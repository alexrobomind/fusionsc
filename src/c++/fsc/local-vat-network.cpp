#include <limits>

#include "local-vat-network.h"

#include <kj/miniposix.h>

namespace {
	
int dupFD(int fd) {
	#if _WIN32
	return _dup(fd);
	#else
	return dup(fd);
	#endif
}

}

namespace fsc {
	
LocalVatNetwork::Message::Message(unsigned int firstSegmentSize) : builder(firstSegmentSize) {}

LocalVatNetwork::Connection::Connection() {}
LocalVatNetwork::Connection::~Connection() {}

lvn::VatId::Reader LocalVatHub::INITIAL_VAT_ID = lvn::INITIAL_VAT_ID.get();

Own<LocalVatHub> newLocalVatHub() {
	return kj::atomicRefcounted<LocalVatHub>();
}

Own<const LocalVatHub> LocalVatHub::addRef() const {
	return kj::atomicAddRef(*this);
}

struct LocalVatNetwork::Connection::IncomingMessage : public capnp::IncomingRpcMessage {
	Message* msg;
	
	IncomingMessage(Message* msg) : msg(msg) {};
	~IncomingMessage() { delete msg; };
	
	capnp::AnyPointer::Reader getBody() override { return msg -> builder.getRoot<capnp::AnyPointer>(); }
	size_t sizeInWords() override { return msg -> builder.sizeInWords(); }
};

struct LocalVatNetwork::Connection::OutgoingMessage : public capnp::OutgoingRpcMessage {
	Connection* conn;
	Message* msg;
	
	OutgoingMessage(Connection& conn, unsigned int firstSegmentWordSize) :
		conn(&conn),
		msg(new Message(firstSegmentWordSize))
	{}
	~OutgoingMessage() { if(msg != nullptr) delete msg; }
	
	capnp::AnyPointer::Builder getBody() override { return msg -> builder.getRoot<capnp::AnyPointer>(); }
	void send() override {
		KJ_IF_MAYBE(ppPeer, conn -> peer) {
			(*ppPeer) -> post(msg);
			msg = nullptr;
		}
	}
	size_t sizeInWords() override { return msg -> builder.sizeInWords(); }
	void setFds(Array<int> fds) override {
		auto builder = kj::heapArrayBuilder<kj::AutoCloseFd>(fds.size());
		for(auto fd : fds) builder.add(dupFD(fd));
		msg -> fds = builder.finish();
	}
};

void LocalVatNetwork::Connection::post(Message* msg) const {
	auto locked = data.lockExclusive();
	
	if(locked -> isClosed)
		return;
	
	locked -> queue.add(*msg);
	
	KJ_IF_MAYBE(ppFulfiller, locked -> readyFulfiller) {
		(*ppFulfiller) -> fulfill();
	}
}

lvn::VatId::Reader LocalVatNetwork::Connection::getPeerVatId() {
	return peerId;
}

Own<capnp::OutgoingRpcMessage> LocalVatNetwork::Connection::newOutgoingMessage(unsigned int firstSegmentSize) {
	return kj::heap<OutgoingMessage>(*this, firstSegmentSize);
}

Promise<Maybe<Own<capnp::IncomingRpcMessage>>> LocalVatNetwork::Connection::receiveIncomingMessage() {	
	auto locked = data.lockExclusive();
	
	auto it = locked -> queue.begin();
	if(it != locked -> queue.end()) {
		Message* msg = &(*it);
		locked -> queue.remove(*it);
		
		Own<capnp::IncomingRpcMessage> result = kj::heap<IncomingMessage>(msg);
		return Maybe<Own<capnp::IncomingRpcMessage>>(mv(result));
	}
	
	if(locked -> isClosed) {
		return Maybe<Own<capnp::IncomingRpcMessage>>(nullptr);
	}

	auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
	locked -> readyFulfiller = mv(paf.fulfiller);
	
	return paf.promise.then([this]() { return receiveIncomingMessage(); });
}

Promise<void> LocalVatNetwork::Connection::shutdown() {
	KJ_IF_MAYBE(ppPeer, peer) {
		auto locked = (*ppPeer) -> data.lockExclusive();
		locked -> isClosed = true;
		
		KJ_IF_MAYBE(ppFulfiller, locked -> readyFulfiller) {
			(*ppFulfiller) -> fulfill();
		}
	}
	
	peer = nullptr;
	return READY_NOW;
}

Own<const LocalVatNetwork::Connection> LocalVatNetwork::Connection::addRef() const {
	return kj::atomicAddRef(*this);
}

lvn::VatId::Reader LocalVatNetwork::getVatId() const {
	return vatId;
}

LocalVatNetwork::LocalVatNetwork(const LocalVatHub& hub) :
	hub(hub.addRef())
{
	auto locked = hub.data.lockExclusive();
	
	KJ_REQUIRE(locked -> freeId < std::numeric_limits<uint64_t>::max());
	uint64_t myID = locked -> freeId++;
	
	locked -> vats.insert(myID, this);
	vatId.setKey(myID);
}

LocalVatNetwork::~LocalVatNetwork() {	
	// De-register vat from the network hub
	{
		auto hubLocked = hub -> data.lockExclusive();
		hubLocked -> vats.erase(vatId.getKey());
	}
	
	// From now on, we will not be receiving accept requests
	// (since they are guarded by the hub lock)
	
	// Clear out the accept queue
	auto locked = data.lockExclusive();
	for(auto& e : locked -> acceptQueue) {
		e.conn -> shutdown();
		locked -> acceptQueue.remove(e);
		delete &e;
	}
	
	KJ_IF_MAYBE(ppFulfiller, locked -> readyFulfiller) {
		(*ppFulfiller) -> reject(KJ_EXCEPTION(DISCONNECTED, "Vat network deleted"));
	}
}

void LocalVatNetwork::acceptPeer(Own<Connection> conn) const {	
	auto e = new AcceptEntry;
	e -> conn = mv(conn);
	
	auto locked = data.lockExclusive();
	locked -> acceptQueue.add(*e);
	
	KJ_IF_MAYBE(ppFulfiller, locked -> readyFulfiller) {
		(*ppFulfiller) -> fulfill();
	}
}

Maybe<Own<LocalVatNetworkBase::Connection>> LocalVatNetwork::connect(lvn::VatId::Reader hostId) {
	auto hubData = hub -> data.lockExclusive();
	
	KJ_IF_MAYBE(pVat, hubData -> vats.find(hostId.getKey())) {
		// Create two connection endpoints
		auto myConn = kj::atomicRefcounted<Connection>();
		auto peerConn = kj::atomicRefcounted<Connection>();
		
		// Link two connections
		myConn -> peer = peerConn -> addRef();
		peerConn -> peer = myConn -> addRef();
		
		// Store peer IDs
		myConn -> peerId.setKey(hostId.getKey());
		peerConn -> peerId.setKey(this -> getVatId().getKey());
		
		// Pass remote end to accept loop
		(*pVat) -> acceptPeer(mv(peerConn));
		return myConn;
	}

	return nullptr;
}

Promise<Own<LocalVatNetworkBase::Connection>> LocalVatNetwork::accept() {
	auto locked = data.lockExclusive();
	
	auto it = locked -> acceptQueue.begin();
	if(it != locked -> acceptQueue.end()) {
		AcceptEntry* e = &(*it);
		Own<LocalVatNetworkBase::Connection> conn = mv(e->conn);
		locked -> acceptQueue.remove(*e);
		delete e;
		return conn;
	}
	
	auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
	locked -> readyFulfiller = mv(paf.fulfiller);
	
	return paf.promise.then([this]() { return accept(); });
}



}