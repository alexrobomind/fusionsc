#include <capi-lvn.h>
#include <capnp/rpc.h>

namespace fsc { namespace capi {
	
namespace {
	// Refcounted message builder
	struct MessageData : public kj::AtomicRefcounted {
		Own<capnp::MessageBuilder> msg;
		Array<kj::ArrayPtr<const capnp::word>> segments = nullptr;
		kj::Array<int> fds = nullptr;
		bool fdsGiven = false;
		
		MessageData(Own<capnp::MessageBuilder> nBuilder) :
			builder(mv(nBuilder))
		{}
		
		~MessageData() {
			if(!fdsGiven) {
				for(auto fd : fds) {
					kj::AutoCloseFd(fd);
				}
			}
		}
	};

	struct MessageImpl : public fusionsc_LvnMessage {
		using VTable = fusionsc_LvnMessage_VTable;
		static VTable V_TABLE;
		
		static VTable createVTable() {
			VTable v;
			v.version = 1;
			
			v.free = [](Message* m) { delete static_cast<MessageImpl*>(m); };
			v.getSegment = [](Message* m, uint32_t segmentId) {
				return static_cast<MessageImpl*>(m) -> getSegment(segmentId);
			};
			v.getFds = [](Message* m) {
				return static_cast<MessageImpl*>(m) -> getFds();
			};
			
			return v;
		};
		
		Own<MessageData> data;
		
		MessageImpl(Own<MessageData> data) : data(mv(data)) {}
		
		SegmentInfo getSegment(uint32_t segmentId) noexcept {			
			if(segmentId >= data -> segments.size()) {
				SegmentInfo result { nullptr, 0 };
				return result;
			}
			
			auto segment = data -> segments[segmentId];
			SegmentInfo result { segment.begin(), segment.size() };
			return result;
		}
		
		FdList getFds() noexcept {
			if(data -> fdsGiven) {
				KJ_LOG(FATAL, "Message::getFds called multiple times");
			}
			
			data -> fdsGiven = true;
			FdList result { data -> fds.begin(), data -> fds.size() };
			return result;
		}
	};

	Message::VTable MessageImpl::V_TABLE = MessageImpl::createVTable();

	capnp::ReaderOptions READ_UNLIMITED {
		kj::maxValue, kj::maxValue
	};

	struct RemoteMessage : public capnp::IncomingRpcMessage {		
		struct Reader : public capnp::MessageReader {
			Message* target;
			
			ArrayPtr<const capnp::word> getSegment(uint32_t segmentId) override {
				Message::SegmentInfo info = target -> v -> getSegment(target, segmentId);
				
				if(info.data == nullptr)
					return nullptr;
				
				return ArrayPtr<const capnp::word>((const capnp::word*) info.data, info.sizeInWords);
			}
			
			Reader(Message* target) : 
				MessageReader(READ_UNLIMITED), // We trust locally received messages
				target(target)
			{}
		};
		
		Reader reader;
		Message* msg;
		kj::Array<kj::AutoCloseFd> fds;
		
		RemoteMessage(Message* m) : reader(m), msg(m) {
			auto remoteFds = m -> v -> getFds(m);
			auto fdBuilder = kj::heapArrayBuilder<kj::AutoCloseFd>(remoteFds.size);
			for(auto i : kj::range(0, remoteFds.size)) {
				fdBuilder.add(remoteFds.data[i]);
			}
			fds = fdBuilder.finish();
		};
		~RemoteMessage() { msg -> v -> free(msg); }
		
		capnp::AnyPointer::Reader getBody() override { return reader.getRoot<capnp::AnyPointer>(); }
		size_t sizeInWords() override { return reader.sizeInWords(); }
		
		kj::ArrayPtr<kj::AutoCloseFd> getAttachedFds() override {
			return fds;
		}
	};

}

}}