#include <limits>

#include "local-vat-network.h"
#include "xthread-queue.h"
#include "data.h"

#include <capnp/rpc.capnp.h>

#include <kj/io.h>

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace fsc { namespace {
	
	struct Message : public fusionsc_LvnMessage {
		std::atomic<uint8_t> refcount = 1;
		
		capnp::MallocMessageBuilder builder;
		
		kj::Array<int> fdStorage;
		kj::Array<fusionsc_SegmentInfo> infoStorage;
		
		bool sent = false;
		
		Message(unsigned int firstSegmentSize) :
			builder(firstSegmentSize)
		{
			free = [](fusionsc_LvnMessage* msg) noexcept {
				auto typed = static_cast<Message*>(msg);
				if(--typed->refcount == 0) {
					delete typed;
				}
			};
		}
		
		~Message() {
			if(!sent) {
				for(auto fd : fdStorage) {
					kj::AutoCloseFd(fd);
				}
			}
		}
		
		Message* prepareForSend() {
			KJ_REQUIRE(!sent);
			sent = true;
			
			auto segments = builder.getSegmentsForOutput();
			
			auto infoBuilder = kj::heapArrayBuilder<fusionsc_SegmentInfo>(segments.size());
			for(auto& segment : segments) {
				fusionsc_SegmentInfo info;
				info.data = segment.begin();
				info.sizeInWords = segment.size();
				infoBuilder.add(info);
			}
			infoStorage = infoBuilder.finish();
			
			// Fill out message fields
			fds = fdStorage.begin();
			fdCount = fdStorage.size();
			
			segmentInfo = infoStorage.begin();
			segmentCount = infoStorage.size();
			
			++refcount;
			return this;
		}
	};
	
	struct LocalVatConnection;
	
	struct OutgoingMessage : public capnp::OutgoingRpcMessage {
		Message* backend;
		Own<LocalVatConnection> conn;
		
		OutgoingMessage(Own<LocalVatConnection>&& nConn, unsigned int firstSegmentSize) :
			backend(new Message(firstSegmentSize)),
			conn(mv(nConn))
		{}
		
		~OutgoingMessage() {
			backend -> free(backend);
		}
		
		capnp::AnyPointer::Builder getBody() override {
			return backend -> builder.getRoot<capnp::AnyPointer>();
		}
		
		void setFds(Array<int> fds) override {
			KJ_REQUIRE(!backend -> sent);
			
			for(auto fd : backend -> fdStorage) {
				kj::AutoCloseFd(fd);
			}
			backend -> fdStorage = mv(fds);
		}
		
		size_t sizeInWords() override {
			return backend -> builder.sizeInWords();
		}
		
		void send() override;
	};
	
	struct IncomingMessage : public capnp::IncomingRpcMessage {
		static constexpr capnp::ReaderOptions READ_UNLIMITED { std::numeric_limits<uint64_t>::max(), std::numeric_limits<int>::max() };
		
		struct MessageReader : public capnp::MessageReader {
			fusionsc_LvnMessage* backend;
			
			MessageReader(fusionsc_LvnMessage* nBackend) :
				capnp::MessageReader(READ_UNLIMITED),
				backend(nBackend)
			{}
			
			ArrayPtr<const capnp::word> getSegment(unsigned int id) override {
				if(id >= backend -> segmentCount)
					return nullptr;
				
				auto& info = backend -> segmentInfo[id];
				return ArrayPtr<const capnp::word>((const capnp::word*) info.data, info.sizeInWords);
			}
		};
		
		fusionsc_LvnMessage* backend;
		MessageReader reader;
		
		kj::Array<kj::AutoCloseFd> ownFds;
		
		IncomingMessage(fusionsc_LvnMessage* nBackend) :
			backend(nBackend),
			reader(nBackend)
		{
			auto fdBuilder = kj::heapArrayBuilder<kj::AutoCloseFd>(backend -> fdCount);
			for(auto i : kj::range(0, backend -> fdCount)) {
				fdBuilder.add(backend -> fds[i]);
			}
			ownFds = fdBuilder.finish();
		}
		
		~IncomingMessage() {
			backend -> free(backend);
		}
		
		capnp::AnyPointer::Reader getBody() override {
			return reader.getRoot<capnp::AnyPointer>();
		}
		
		kj::ArrayPtr<kj::AutoCloseFd> getAttachedFds() override {
			return ownFds;
		}
		
		size_t sizeInWords() override {
			return reader.sizeInWords();
		}
	};
	
	struct LocalEndPoint : public fusionsc_LvnEndPoint {
		std::atomic<uint64_t> refcount = 2; // This class is always created with 2 refs initially
		XThreadQueue<Own<capnp::IncomingRpcMessage>> queue;
		
		LocalEndPoint() {
			incRef = [](fusionsc_LvnEndPoint* t) noexcept {
				auto typed = static_cast<LocalEndPoint*>(t);
				++(typed -> refcount);
			};
			
			decRef = [](fusionsc_LvnEndPoint* t) noexcept {
				auto typed = static_cast<LocalEndPoint*>(t);
				if(--(typed -> refcount) == 0) {
					delete typed;
				}
			};
			
			receive = [](fusionsc_LvnEndPoint* t, fusionsc_LvnMessage* msg) noexcept {
				auto typed = static_cast<LocalEndPoint*>(t);
				typed -> queue.push(kj::heap<IncomingMessage>(msg));
			};
			
			close = [](fusionsc_LvnEndPoint* t) noexcept {
				auto typed = static_cast<LocalEndPoint*>(t);
				typed -> queue.close(KJ_EXCEPTION(DISCONNECTED, "Remote end closed"));
			};
		}
	};
	
	struct LocalVatConnection : public LocalVatNetworkBase::Connection, public kj::Refcounted {
		LocalEndPoint* localEndPoint;
		fusionsc_LvnEndPoint* remoteEndPoint;
		
		Temporary<lvn::VatId> peerId;
		
		LocalVatConnection() :
			localEndPoint(nullptr),
			remoteEndPoint(nullptr)
		{}
		
		~LocalVatConnection() {
			if(localEndPoint != nullptr) {
				localEndPoint -> queue.close();
				localEndPoint -> queue.clear();
				localEndPoint -> decRef(localEndPoint);
			}
			
			if(remoteEndPoint != nullptr)
				remoteEndPoint -> decRef(remoteEndPoint);
		}
		
		Own<capnp::OutgoingRpcMessage> newOutgoingMessage(unsigned int firstSegmentSize) override {
			return kj::heap<OutgoingMessage>(kj::addRef(*this), firstSegmentSize);
		}
		
		Promise<Maybe<Own<capnp::IncomingRpcMessage>>> receiveIncomingMessage() override {
			using MaybeMsg = Maybe<Own<capnp::IncomingRpcMessage>>;
			
			return localEndPoint -> queue.pop().then(
				[](auto msg) -> MaybeMsg { return mv(msg); },
				[](auto&& exc) -> MaybeMsg { return nullptr; }
			);
		}
		
		lvn::VatId::Reader getPeerVatId() override {
			return peerId.asReader();
		}
		
		Promise<void> shutdown() override {
			localEndPoint -> queue.close();
			localEndPoint -> queue.clear();
			
			remoteEndPoint -> close(remoteEndPoint);
			
			return READY_NOW;
		}
	};
	
	struct AcceptRequest {
		uint64_t peerId;
		fusionsc_LvnEndPoint* remoteEndPoint;
		LocalEndPoint* localEndPoint;
		
		~AcceptRequest() {
			if(remoteEndPoint != nullptr)
				remoteEndPoint -> decRef(remoteEndPoint);
			
			if(localEndPoint != nullptr) {
				localEndPoint -> queue.close();
				localEndPoint -> queue.clear();
				localEndPoint -> decRef(localEndPoint);
			}
		}
	};
	
	struct LvnAcceptListener : public fusionsc_LvnListener {
		std::atomic<uint64_t> refcount = 1;
		
		XThreadQueue<Own<AcceptRequest>> queue;
		
		LvnAcceptListener() {
			free = [](fusionsc_LvnListener* t) {
				auto typed = static_cast<LvnAcceptListener*>(t);
				
				if(--(typed -> refcount) == 0)
					delete typed;
			};
			
			accept = [](fusionsc_LvnListener* t, uint64_t peerId, fusionsc_LvnEndPoint* peer) -> fusionsc_LvnEndPoint* {
				auto typed = static_cast<LvnAcceptListener*>(t);
				
				auto ar = kj::heap<AcceptRequest>();
				ar -> peerId = peerId;
				ar -> remoteEndPoint = peer;
				
				auto lep = new LocalEndPoint();
				ar -> localEndPoint = lep;
				
				typed -> queue.push(mv(ar));
				return lep;
			};
		}
	};
		
	void OutgoingMessage::send() {
		conn -> remoteEndPoint -> receive(conn -> remoteEndPoint, backend -> prepareForSend());
	}
} // Anonymous namespace
	
struct LocalVatNetwork::Impl {
	LvnAcceptListener* listener;
	fusionsc_Lvn* backend;
	Temporary<lvn::VatId> vatId;
	
	Impl(LvnHub& hub) : listener(new LvnAcceptListener()) {
		fusionsc_LvnHub* cHub = hub.get();
		
		++(listener -> refcount);
		backend = cHub -> join(cHub, listener);
		vatId.setKey(backend -> address);
	}
	
	~Impl() {
		backend -> decRef(backend);
		
		listener -> queue.close();
		listener -> queue.clear();
		listener -> free(listener);
	}
};

Maybe<Own<LocalVatNetworkBase::Connection>> LocalVatNetwork::connect(lvn::VatId::Reader hostId) {
	auto connection = kj::refcounted<LocalVatConnection>();
	
	auto lep = new LocalEndPoint();
	connection -> localEndPoint = lep;
	fusionsc_LvnEndPoint* remoteEnd = pImpl -> backend -> connect(pImpl -> backend, hostId.getKey(), lep);
	
	if(remoteEnd == nullptr)
		return nullptr;
	
	connection -> remoteEndPoint = remoteEnd;
	connection -> peerId.setKey(hostId.getKey());
	
	return connection;
}

Promise<Own<LocalVatNetworkBase::Connection>> LocalVatNetwork::accept() {
	return pImpl -> listener -> queue.pop()
	.then([](Own<AcceptRequest> request) -> Own<LocalVatNetworkBase::Connection> {
		auto connection = kj::refcounted<LocalVatConnection>();
		connection -> localEndPoint = request -> localEndPoint;
		connection -> remoteEndPoint = request -> remoteEndPoint;
		connection -> peerId.setKey(request -> peerId);
		
		request -> localEndPoint = nullptr;
		request -> remoteEndPoint = nullptr;
		
		return connection;
	});
}

lvn::VatId::Reader LocalVatNetwork::getVatId() const {
	return pImpl -> vatId;
}

LocalVatNetwork::LocalVatNetwork(LvnHub& hub) :
	pImpl(kj::heap<Impl>(hub))
{}

LocalVatNetwork::~LocalVatNetwork() {}


}