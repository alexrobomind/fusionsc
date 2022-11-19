#include "local-vat-network.h"

namespace fsc { namespace lvn {

using LocalVatNetworkBase = capnp::TwoPartyVatNetworkBase;
using LocalMessage = Own<capnp::MallocMessageBuilder>;

using VatId = capnp::rpc::twoparty::VatId;

namespace {
	
struct MessageBlock {
	capnp::MallocMessageBuilder builder;
	kj::Array<int> fds;
	kj::ListLink<MessageBlock> link;
	
	MessageBlock(unsigned int fsSize) : builder(fsSize) {}
};

struct LocalIncomingMessage : public IncomingRpcMessage {
	MessageBlock* block;
	
	LocalIncomingMessage(MessageBlock* block) : block(block) {};
	~LocalIncomingMessage() { delete block; };
	
	inline AnyPointer::Reader getBody() override { return block -> builder.getRoot<AnyPointer>(); }
	inline size_t sizeInWords() override { return block -> builder.sizeInWords(); }
	
	kj::ArrayPtr<kj::AutoCloseFd> getAttachedFds() override;
};

struct LocalConnection : public LocalVatNetworkBase::Connection, public kj::AtomicRefcounted {
	struct Data {
		bool isClosed = false;
		kj::List<MessageBlock> blocks;
	
		Own<kj::CrossThreadPromiseFulfiller<void>> readyFulfiller;
	};
	
	~LocalConnection() {
		auto locked = data.lockExclusive();
		
		for(auto& block : blocks) {
			delete &block;
		}
	}
	
	void close() {
		auto locked = data.lockExclusive();
		
		locked -> isClosed = true;
		for(auto& block : blocks) {
			delete &block;
		}
	}	
		
	Own<const LocalConnection> peer;
	
	MutexGuarded<Data> data;
	MallocMessageBuilder peerId;
	
	::vsc::lvn::VatId::Reader getPeerVatId() override;
	
	inline void send(MessageBlock* block) {
		auto locked = peer->data.lockExclusive();
		
		if(locked -> isClosed)
			return;
		
		locked -> blocks.add(*block);
		
		if(locked -> readyFulfiller != nullptr)
			locked -> readyFulfiller -> fulfill();
	}
	
	kj::Promise<kj::Maybe<kj::Own<IncomingRpcMessage>>> receiveIncomingMessage() override {		
		auto locked = data.lockExclusive();
		
		auto it = locked -> blocks.begin();
		if(it != locked -> blocks.end()) {
			MessageBlock* block = &(*it);
			locked -> blocks.remove(*it);
			
			return kj::heap<LocalIncomingMessage>(block);
		}
		
		if(locked -> isClosed)
			return nullptr;
	
		auto paf = kj::newPromiseAndCrossThreadFulfiller();
		locked -> readyFulfiller = mv(paf.fulfiller);
		
		return paf.promise.then([this]() { return receiveIncomingMessage(); });
	}
	
	kj::Own<OutgoingRpcMessage> newOutgoingMessage(unsigned int firstSegmentWordSize) override; // Implemented below
	
	kj::Promise<void> shutdown() override {	
		auto locked = peer -> data.lockExclusive();
		locked.isClosed = true;
		
		return READY_NOW;
	}
		
};

struct LocalOutgoingMessage : public OutgoingRpcMessage {
	LocalConnection* conn;
	MessageBlock* block;
	
	LocalOutgoingMessage(MessageBlock* block) : block(block) {}
	inline ~LocalOutgoingMessage() { if(block != nullptr) delete block;	}
	
	inline AnyPointer::Builder getBody() override { return block -> builder.getRoot<AnyPointer>(); }
	inline void send() override { conn -> send(block); block = nullptr;	}
	inline size_t sizeInWords() override { return block -> builder.sizeInWords(); }
	
	void setFds(kj::Array<int> fds) override;
};

kj::Own<OutgoingRpcMessage> LocalConnection::newOutgoingMessage(unsigned int firstSegmentWordSize) {
	MessageBlock* mb = new MessageBlock(firstSegmentWordSize);
	return kj::heap<LocalOutgoingMessage>(mb);
}

struct ConnectionRequest {
	Own<kj::CrossThreadPromiseFulfiller<Own<LocalConnection>>> putConnectionHere;
	Own<LocalConnection> unfinishedConnection;
	ListLink<ConnectionRequest> pendingRequests;
};

class LocalVatNetworkImpl : public LocalVatNetwork {
	
};

}

}}