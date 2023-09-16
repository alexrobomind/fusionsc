#include <limits>

#include "local-vat-network.h"
#include "xthread-queue.h"

#include <capnp/rpc.capnp.h>

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace fsc { namespace {
	int dupFD(int fd) {
		#if _WIN32
		return _dup(fd);
		#else
		return dup(fd);
		#endif
	}
	
	struct Message : kj::AtomicRefcounted {
		capnp::MallocMessageBuilder builder;
		kj::ListLink<Message> link;
		kj::Array<int> fds;
		bool fdsTaken = false;
		
		Message(unsigned int firstSegmentSize) :
			builder(firstSegmentSize)
		{}
		
		~Message() {
			if(!fdsTaken) {
				for(auto fd : fds) {
					kj::AutoCloseFd afd(fd);
				}
			}
		}
		
		Own<Message> addRef() {
			return kj::atomicAddRef(*this);
		}
	};
	
	struct ExchangePoint : public kj::AtomicRefcounted {
		XThreadQueue<Own<capnp::IncomingRpcMessage>> queue;
	};

	struct Connection : public LocalVatNetworkBase::Connection {
		Connection(Own<ExchangePoint> incoming, Own<ExchangePoint> outgoing):
			incoming(mv(incoming)),
			outgoing(mv(outgoing))
		{};
		
		~Connection() { shutdown(); };
			
		lvn::VatId::Reader getPeerVatId() override {
			return peerId;
		}
		
		Own<capnp::OutgoingRpcMessage> newOutgoingMessage(unsigned int firstSegmentSize) override {
			return kj::heap<OutgoingMessage>(*this, firstSegmentSize);
		}
		
		Promise<Maybe<Own<capnp::IncomingRpcMessage>>> receiveIncomingMessage() override;
		Promise<void> shutdown() override {
			incoming -> queue.close();
			incoming -> queue.clear();
			outgoing -> queue.close(KJ_EXCEPTION(DISCONNECTED, "Remote end closed"));
			
			return READY_NOW;
		}
		
		Own<ExchangePoint> incoming;
		Own<ExchangePoint> outgoing;
		
		Temporary<lvn::VatId> peerId;

		struct IncomingMessage : public capnp::IncomingRpcMessage {
			Own<Message> msg;
			kj::Array<kj::AutoCloseFd> fds;
			
			IncomingMessage(Own<Message>&& msgParam) :
				msg(mv(msgParam))
			{
				auto fdsBuilder = kj::heapArrayBuilder<kj::AutoCloseFd>(msg -> fds.size());
				for(auto fd : msg -> fds)
					fdsBuilder.add(fd);
				fds = fdsBuilder.finish();
				
				msg -> fdsTaken = true;
			}
			
			capnp::AnyPointer::Reader getBody() override { return msg -> builder.getRoot<capnp::AnyPointer>(); }
			size_t sizeInWords() override { return msg -> builder.sizeInWords(); }
		};

		struct OutgoingMessage : public capnp::OutgoingRpcMessage {
			Connection* conn;
			Own<Message> msg;
			
			OutgoingMessage(Connection& conn, unsigned int firstSegmentWordSize) :
				conn(&conn),
				msg(kj::atomicRefcounted<Message>(firstSegmentWordSize))
			{}
			
			capnp::AnyPointer::Builder getBody() override { return msg -> builder.getRoot<capnp::AnyPointer>(); }
			void send() override {
				conn -> outgoing -> queue.push(kj::heap<IncomingMessage>(msg -> addRef()));
			}
			
			size_t sizeInWords() override { return msg -> builder.sizeInWords(); }
			void setFds(Array<int> fds) override {
				if(!msg -> fdsTaken) {
					for(auto fd : msg -> fds) {
						kj::AutoCloseFd afd(fd);
					}
				}
				
				msg -> fds = mv(fds);
				msg -> fdsTaken = false;
			}
		};
	};
}

struct LocalVatNetwork::Impl {		
	Own<const LocalVatHub> hub;
	Temporary<lvn::VatId> vatId;
	XThreadQueue<Own<Connection>> acceptQueue;
	
	Impl(const LocalVatHub& paramHub, LocalVatNetwork& parent) :
		hub(paramHub.addRef())
	{
		auto locked = hub -> data.lockExclusive();
		
		KJ_REQUIRE(locked -> freeId < std::numeric_limits<uint64_t>::max());
		uint64_t myID = locked -> freeId++;
		
		locked -> vats.insert(myID, &parent);
		vatId.setKey(myID);
	}
	
	~Impl() {
		// De-register vat from the network hub
		{
			auto hubLocked = hub -> data.lockExclusive();
			hubLocked -> vats.erase(vatId.getKey());
		}
		
		// From now on, we will not be receiving accept requests
		// (since they are guarded by the hub lock)		
		acceptQueue.close(KJ_EXCEPTION(DISCONNECTED, "Vat network deleted"));
	}
};

lvn::VatId::Reader LocalVatHub::INITIAL_VAT_ID = lvn::INITIAL_VAT_ID.get();

Own<LocalVatHub> newLocalVatHub() {
	return kj::atomicRefcounted<LocalVatHub>();
}

Own<const LocalVatHub> LocalVatHub::addRef() const {
	return kj::atomicAddRef(*this);
}

Promise<Maybe<Own<capnp::IncomingRpcMessage>>> Connection::receiveIncomingMessage() {
	using MaybeMsg = Maybe<Own<capnp::IncomingRpcMessage>>;
	
	return incoming -> queue.pop().then(
		[](auto msg) -> MaybeMsg { return mv(msg); },
		[](auto&& exc) -> MaybeMsg { return nullptr; }
	);
}

lvn::VatId::Reader LocalVatNetwork::getVatId() const {
	return pImpl -> vatId;
}

LocalVatNetwork::LocalVatNetwork(const LocalVatHub& hub) :
	pImpl(kj::heap<Impl>(hub, *this))
{}

LocalVatNetwork::~LocalVatNetwork() {}

Maybe<Own<LocalVatNetworkBase::Connection>> LocalVatNetwork::connect(lvn::VatId::Reader hostId) {
	auto hubData = pImpl -> hub -> data.lockExclusive();
	
	KJ_IF_MAYBE(pVat, hubData -> vats.find(hostId.getKey())) {
		// Create two connection endpoints
		auto ex1 = kj::atomicRefcounted<ExchangePoint>();
		auto ex2 = kj::atomicRefcounted<ExchangePoint>();
		
		auto myConn = kj::heap<::fsc::Connection>(kj::atomicAddRef(*ex1), kj::atomicAddRef(*ex2));
		auto peerConn = kj::heap<::fsc::Connection>(mv(ex2), mv(ex1));
		
		// Store peer IDs
		myConn -> peerId.setKey(hostId.getKey());
		peerConn -> peerId.setKey(this -> getVatId().getKey());
		
		// Pass remote end to accept loop
		(*pVat) -> pImpl -> acceptQueue.push(mv(peerConn));
		return myConn;
	}

	return nullptr;
}

Promise<Own<LocalVatNetworkBase::Connection>> LocalVatNetwork::accept() {
	return pImpl -> acceptQueue.pop();
}

}