#include <list>

#include "local.h"
#include "data.h"

namespace fsc {
	
// === class LibraryHandle ===

void LibraryHandle::stopSteward() const {
	KJ_LOG(WARNING, "LibraryHandle::stopSteward()");
	_daemonRunner->disconnect();
	KJ_LOG(WARNING, "DaemonRunner disconnected");
	storeSteward.stop();
}
	
// === class ThreadHandle ===

struct ThreadHandle::Ref {		
	Ref(const kj::MutexGuarded<RefData>* refData);
	~Ref();
		
	kj::ListLink<Ref> link;
	const kj::MutexGuarded<RefData>* refData;
};

struct ThreadHandle::RefData {
	kj::List<Ref, &Ref::link> refs;
	Own<CrossThreadPromiseFulfiller<void>> whenNoRefs;
	const ThreadHandle* owner;
};
		
ThreadHandle::Ref::Ref(const kj::MutexGuarded<RefData>* refData) :
	refData(refData)
{
	refData -> lockExclusive() -> refs.add(*this);
}

ThreadHandle::Ref::~Ref() {
	auto locked = refData -> lockExclusive();
	locked -> refs.remove(*this);
	
	auto& pFulfiller = locked -> whenNoRefs;
	if(pFulfiller.get() != nullptr) {
		pFulfiller->fulfill();
		pFulfiller = nullptr;
	}
}

ThreadHandle::ThreadHandle(Library l) :
	_ioContext(kj::setupAsyncIo()),
	_library(l -> addRef()),
	_executor(kj::getCurrentThreadExecutor()),
	_dataService(kj::heap<LocalDataService>(l)),
	_filesystem(kj::newDiskFilesystem()),
	refData(new kj::MutexGuarded<RefData>())
{	
	KJ_REQUIRE(current == nullptr, "Can only have one active ThreadHandle / LibraryThread per thread");
	current = this;
}

ThreadHandle::~ThreadHandle() {
	KJ_REQUIRE(current == this, "Destroying LibraryThread in wrong thread") {}
	
	// If the library is in shutdown mode, we have to (conservatively) assume 
	// that all other threads  have unexpectedly died and this is the last
	// remaining active thread. We can not BANK on it, but we can not wait
	// for any other promises on the event loop to resolve.
	if(!_library->inShutdownMode()) {
		while(true) {
			Promise<void> noMoreRefs = READY_NOW;
			
			{
				auto locked = refData -> lockExclusive();
				
				if(locked -> refs.size() == 0)
					break;
				
				auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
				noMoreRefs = mv(paf.promise);
				
				locked -> whenNoRefs = mv(paf.fulfiller);
			}
			
			noMoreRefs.wait(waitScope());
		}
		
		delete refData;
	} else {
		bool canDeleteRefdata = false;
		{
			auto locked = refData -> lockExclusive();
		
			if(locked -> refs.size() == 0)
				canDeleteRefdata = true;
		}
		
		if(canDeleteRefdata)
			delete refData;
	}
			
	current = nullptr;
}

Own<ThreadHandle> ThreadHandle::addRef() {
	return Own<ThreadHandle>(this, kj::NullDisposer::instance).attach(kj::heap<ThreadHandle::Ref>(this->refData));
}

Own<const ThreadHandle> ThreadHandle::addRef() const {
	return Own<const ThreadHandle>(this, kj::NullDisposer::instance).attach(kj::heap<ThreadHandle::Ref>(this->refData));
}

// === class CrossThreadConnection ===

CrossThreadConnection::CrossThreadConnection():
	_exec(kj::getCurrentThreadExecutor()),
	_port(kj::newPromiseAndFulfiller<kj::uint>()),
	_acceptCalled(false),
	_connectCalled(false)
{}

Promise<Own<kj::AsyncIoStream>> CrossThreadConnection::accept(ThreadHandle& h) {
	// Make sure this method is only called once
	KJ_REQUIRE(!_acceptCalled, "CrossThreadConnection::accept(...) may only be called once");
	_acceptCalled = true;
	
	// Prepare a random security tokens
	h.rng().randomize((ArrayPtr<byte>) sendToken1);
	h.rng().randomize((ArrayPtr<byte>) sendToken2);
		
	// Restrict connections to local device
	auto restrictedNetwork = h.network().restrictPeers({"local"});
	auto parseAddrOp = restrictedNetwork -> parseAddress("127.0.0.1");
	return parseAddrOp.then([this](Own<kj::NetworkAddress> addr){
		KJ_REQUIRE(addr.get() != nullptr);
		
		// Start listening on an assigned port
		auto recv = addr -> listen();
		KJ_REQUIRE(recv.get() != nullptr);
		
		// Store the port we are listening on in the fulfiller
		auto port = recv -> getPort();
		auto portWriteOp = _exec.executeAsync([this, port]() mutable { _port.fulfiller -> fulfill(mv(port)); });
		
		// Accept a single stream from the receiver, then close it
		return portWriteOp.then([recv = mv(recv), addr = mv(addr)]() mutable {
			return recv -> accept().attach(mv(recv),mv(addr));
		});
	})
	.attach(mv(restrictedNetwork))
	.then([this](Own<kj::AsyncIoStream> stream) {
		// After accepting the stream, read the security token posted by the other side
		KJ_REQUIRE(stream.get() != nullptr);
		auto ops = stream -> read(recvToken1.begin(), recvToken1.size())
		.then([this] () {
			KJ_REQUIRE((ArrayPtr<byte>) recvToken1 == (ArrayPtr<byte>) sendToken1, "Security token mismatch");
		})
		.then([this, &stream = *stream] () {
			return stream.write(sendToken2.begin(), sendToken2.size());
		});
		
		return ops.then([stream2 = mv(stream)] () mutable {
			return mv(stream2);
		});
		
		//return readOp.then(mv(checkFun)).then(mv(writeFun)).then(mv(returnStream));
	});
}

Promise<Own<kj::AsyncIoStream>> CrossThreadConnection::connect(ThreadHandle& h) {
	// Make sure this method is only called once
	KJ_REQUIRE(!_connectCalled, "CrossThreadConnection::connect(...) may only be called once");
	_connectCalled = true;
	
	// Read the port number (also waits async until the listener is open)
	return _exec.executeAsync([this]() { return mv(_port.promise); }) // Retrieve port number
	.then([&](kj::uint portNo) {
		// Prepare network address
		return h.network().parseAddress("127.0.0.1", portNo);
	})
	.then([](Own<kj::NetworkAddress> addr) {
		KJ_REQUIRE(addr.get() != nullptr);
		
		// Connect to target port
		return addr -> connect().attach(mv(addr));
	})
	.then([this](Own<kj::AsyncIoStream> stream) {
		KJ_REQUIRE(stream.get() != nullptr);
		
		// After connecting, write the security token
		auto ops = stream -> write(sendToken1.begin(), sendToken1.size())
		
		// Now read the response token
		.then([this, &stream = *stream] () {
			return stream.read(recvToken2.begin(), recvToken2.size());
		})
		
		// Check that it is OK
		.then([this] () {
			KJ_REQUIRE((ArrayPtr<byte>) recvToken2 == (ArrayPtr<byte>) sendToken2, "Security token mismatch");
		});
		
		// Finally, return the stream
		return ops.then([stream2 = mv(stream)] () mutable {
			return mv(stream2);
		});
	})
	;
}

// ============================ class DaemonRunner ===========================

namespace {
	class TaskSetErrorHandler : public kj::TaskSet::ErrorHandler {
		void taskFailed(kj::Exception&& exception) override {
			KJ_LOG(WARNING, "Exception in daemon task", exception);
		}
	};
	
	TaskSetErrorHandler errorHandler;
}

DaemonRunner::DaemonRunner(const kj::Executor& executor) {
	executor.executeSync([this, &executor]() {
		auto locked = connection.lockExclusive();
		
		*locked = Connection { executor.addRef(), kj::heap<kj::TaskSet>(errorHandler) };
	});
}

bool DaemonRunner::run(kj::Function<Promise<void>()> func) const {
	auto locked = connection.lockExclusive();
	KJ_IF_MAYBE(pConn, *locked) {
		pConn -> executor -> executeSync([func = mv(func), &tasks = *(pConn -> tasks)]() mutable {
			Promise<void> task = kj::evalLater(mv(func));
			tasks.add(mv(task));
		});
		
		return true;
	}
	
	return false;
}

void DaemonRunner::disconnect() const {
	auto locked = connection.lockExclusive();
	
	KJ_IF_MAYBE(pConn, *locked) {
		Own<const kj::Executor> executor = mv(pConn -> executor);
		
		try {
			executor -> executeSync([&locked]() {
				*locked = nullptr;
			});
		} catch(kj::Exception e) {
			KJ_LOG(WARNING, "Exception in cleanup routine", e);
		}
	}
}

Promise<void> DaemonRunner::whenDone() const {
	auto locked = connection.lockExclusive();
	
	KJ_IF_MAYBE(pConn, *locked) {
		return pConn -> executor -> executeAsync([this]() -> Promise<void> {
			// Since we are wrapped inside executeAsync, this is guaranteed not to block
			// with the outer lock
			auto locked = connection.lockExclusive();
			
			KJ_IF_MAYBE(pConn, *locked) {
				return pConn -> tasks -> onEmpty();
			}
			return READY_NOW;
		}).attach(addRef());
	}
	
	return READY_NOW;
}

// ======================= crossThreadPipe ================================

namespace {
	struct AsyncMessageQueue : kj::AtomicRefcounted {		
		struct DstBufferQueueEntry {
			kj::ListLink<DstBufferQueueEntry> link;
			kj::ArrayPtr<byte> ptr;
			Own<PromiseFulfiller<size_t>> whenConsumed;
			size_t minBytes;
		};
		
		struct SrcBufferQueueEntry {
			kj::ListLink<SrcBufferQueueEntry> link;
			kj::ArrayPtr<const byte> ptr;
			Maybe<Own<CrossThreadPromiseFulfiller<void>>> whenConsumed;
		};
		
		//! Members shared across different threads
		struct Shared {
			kj::List<SrcBufferQueueEntry, &SrcBufferQueueEntry::link> queue;
			Own<CrossThreadPromiseFulfiller<void>> newInputsReady;
			
			Maybe<kj::Exception> sendError;
		};
		
		//! Members local to the reader thread
		struct ReaderPrivate {
			kj::List<SrcBufferQueueEntry, &SrcBufferQueueEntry::link> inQueue;
			kj::List<DstBufferQueueEntry, &DstBufferQueueEntry::link> outQueue;

			size_t inConsumed;
			size_t outConsumed;
			
			Promise<void> newInputsReady = READY_NOW;
					
			Maybe<Promise<void>> pumpLoop;
			Maybe<kj::Exception> readError;
			kj::Canceler readCanceler;
		};
		
		Own<const AsyncMessageQueue> addRef() const { return kj::atomicAddRef(*this); }
		
		Promise<void> write(ArrayPtr<const ArrayPtr<const byte>> queue) const {
			auto locked = shared.lockExclusive();
			
			KJ_IF_MAYBE(pErr, locked->sendError) {
				return cp(*pErr);
			}
			
			for(auto ptr : queue) {
				SrcBufferQueueEntry* e = new SrcBufferQueueEntry();
				e -> ptr = ptr;
				locked -> queue.add(*e);
			}
			
			auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
			SrcBufferQueueEntry* barrier = new SrcBufferQueueEntry();
			barrier -> whenConsumed = mv(paf.fulfiller);
			locked -> queue.add(*barrier);
		
			if(locked -> newInputsReady.get() != nullptr)
				locked -> newInputsReady -> fulfill();
			
			return mv(paf.promise);
		}
		
		void shutdownWriteEnd(kj::Exception e) const {
			auto locked = shared.lockExclusive();
			
			locked -> sendError = e;
			
			if(locked -> newInputsReady.get() != nullptr)
				locked -> newInputsReady->reject(cp(e));
			
			auto& queue = locked -> queue;
			
			for(auto& eIn : queue) {
				queue.remove(eIn);
					
				KJ_IF_MAYBE(pWC, eIn.whenConsumed) {
					(**pWC).reject(cp(e));
				}
				
				delete &eIn;
			}
		}
		
		Promise<size_t> read(kj::ArrayPtr<byte> out, size_t minBytes) {
			auto& mine = readerPrivate;
			
			auto paf = kj::newPromiseAndFulfiller<size_t>();
			
			auto e = new DstBufferQueueEntry();
			e -> ptr = out;
			e -> minBytes = minBytes;
			e -> whenConsumed = mv(paf.fulfiller);
			
			mine.outQueue.add(*e);
			
			startPumpLoop();
			
			return mv(paf.promise);
		}
		
		void shutdownReadEnd(kj::Exception&& reason) {		
			// Stop pump loop
			readerPrivate.readCanceler.cancel(mv(reason));
			readerPrivate.readError = reason;
			readerPrivate.pumpLoop = NEVER_DONE;
			
			// Perform last pump
			pumpLocal();
			
			// Delete existing fulfiller
			auto locked = shared.lockExclusive();
			
			if(locked -> newInputsReady.get() != nullptr) {
				KJ_DBG("Clear NIR", this);
				locked -> newInputsReady -> reject(cp(reason));
				locked -> newInputsReady = nullptr;
			}
		}
		
		Maybe<uint64_t> tryGetLength() {
			auto locked = shared.lockExclusive();
			
			// We can only predict the no. of remaining elements if the send end is shut down
			if(locked -> sendError == nullptr)
				return nullptr;
			
			if(readerPrivate.outQueue.size() != 0)
				return nullptr;
			
			uint64_t total = 0;
			for(auto& e : locked -> queue)
				total += e.ptr.size();
			
			for(auto& e : readerPrivate.inQueue)
				total += e.ptr.size();
			
			total -= readerPrivate.inConsumed;
			
			KJ_REQUIRE(readerPrivate.outConsumed == 0, "Internal error");
			
			return total;
		}
		
	private:
		kj::MutexGuarded<Shared> shared;
		ReaderPrivate readerPrivate;
		
		bool pumpLocal() {
			KJ_DBG("pumpLocal");
			auto& mine = readerPrivate;
			
			while(true) {
				if(mine.inQueue.size() == 0 || mine.outQueue.size() == 0)
					break;
				
				auto& eIn = *mine.inQueue.begin();
				auto& eOut = *mine.outQueue.begin();
				
				auto inBuf = eIn.ptr;
				auto outBuf = eOut.ptr;
				
				size_t inRem = inBuf.size() - mine.inConsumed;
				size_t outRem = outBuf.size() - mine.outConsumed;
				
				size_t rem = std::min(inRem, outRem);
				
				KJ_DBG("transfer", rem);
				
				if(rem > 0)
					memcpy(outBuf.begin() + mine.outConsumed, inBuf.begin() + mine.inConsumed, rem);
				
				mine.inConsumed += rem;
				mine.outConsumed += rem;
				
				if(mine.inConsumed >= inBuf.size()) {
					mine.inQueue.remove(eIn);
					
					KJ_IF_MAYBE(pWC, eIn.whenConsumed) {
						(**pWC).fulfill();
					}
					
					delete &eIn;
					mine.inConsumed = 0;
				}
								
				if(mine.outConsumed == outBuf.size()) {
					mine.outQueue.remove(eOut);
					eOut.whenConsumed -> fulfill(cp(mine.outConsumed));
					
					delete &eOut;
					mine.outConsumed = 0;
				}
			}
			
			// If there are outstanding partial reads, fulfill them if possible
			// OR if the read end has shut down
			while(mine.outQueue.size() > 0) {
				auto& eOut = *mine.outQueue.begin();
				
				if(mine.outConsumed >= eOut.minBytes || mine.readError != nullptr) {
					mine.outQueue.remove(eOut);
					eOut.whenConsumed -> fulfill(cp(mine.outConsumed));
					
					delete &eOut;
					mine.outConsumed = 0;
				} else {
					break;
				}
			}
			
			// If there are empty writes (barrier entries), fulfill them
			while(mine.inQueue.size() > 0) {
				auto& eIn = *mine.inQueue.begin();
				
				if(mine.inConsumed == eIn.ptr.size()) {
					mine.inQueue.remove(eIn);
					
					KJ_IF_MAYBE(pWC, eIn.whenConsumed) {
						(**pWC).fulfill();
					}
					
					delete &eIn;
					mine.inConsumed = 0;
				} else {
					break;
				}
			}
			
			return mine.outQueue.size() == 0;
		}
		
		void steal() {
			auto locked = shared.lockExclusive();
			auto& mine = readerPrivate;
			
			for(auto& e : locked -> queue) {
				locked -> queue.remove(e);
				mine.inQueue.add(e);
			}
			
			KJ_IF_MAYBE(pErr, locked->sendError) {
				mine.newInputsReady = cp(*pErr);
			} else {
				KJ_DBG("Set NIR", this);
				auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
				locked -> newInputsReady = mv(paf.fulfiller);
				mine.newInputsReady = mv(paf.promise);
			}
		}
		
		void startPumpLoop() {
			auto& mine = readerPrivate;
			
			if(mine.pumpLoop != nullptr)
				return;
			
			mine.pumpLoop = mine.readCanceler.wrap(pump())/*
			.eagerlyEvaluate([this, &mine](kj::Exception e){
				mine.readError = e;
				pumpLocal();
			})*/;
		}
		
		Promise<void> pump() {
			steal();
			pumpLocal();
			
			return readerPrivate.newInputsReady.then([this]() { return pump(); });
		}
	};
	
	struct MessageQueueIOStream : public kj::AsyncIoStream {
		Own<AsyncMessageQueue> readFrom;
		Own<const AsyncMessageQueue> writeTo;
		
		MessageQueueIOStream(Own<AsyncMessageQueue> readFromIn, Own<const AsyncMessageQueue> writeToIn):
			readFrom(mv(readFromIn)),
			writeTo(mv(writeToIn))
		{}
		
		~MessageQueueIOStream() {
			writeTo  -> shutdownWriteEnd(KJ_EXCEPTION(DISCONNECTED, "Stream destroyed"));
			readFrom -> shutdownReadEnd(KJ_EXCEPTION(DISCONNECTED, "Stream destroyed"));
		}
		
		// ======== AsyncOutputStream =========
		
		Promise<void> write(const void* buffer, size_t size) override {
			KJ_STACK_ARRAY(ArrayPtr<const byte>, out, 1, 1, 1);
			
			out[0] = ArrayPtr(reinterpret_cast<const byte*>(buffer), size);
			return writeTo -> write(out);
		}
		
		Promise<void> write(ArrayPtr<const ArrayPtr<const byte>> pieces) override {
			return writeTo -> write(pieces);
		}
		
		Promise<void> whenWriteDisconnected() override { return NEVER_DONE; }
		
		// ======== AsyncInputStream =========
		
		Promise<size_t> tryRead(void* buffer, size_t minBytes, size_t maxBytes) override {
			kj::ArrayPtr<byte> bufPtr(
				reinterpret_cast<byte*>(buffer),
				maxBytes
			);
			return readFrom -> read(bufPtr, minBytes);
		}
		
		Maybe<uint64_t> tryGetLength() override {
			return readFrom -> tryGetLength();
		}
		
		// ========= AsyncIoStream ===========
		
		void shutdownWrite() override {
			writeTo -> shutdownWriteEnd(KJ_EXCEPTION(DISCONNECTED, "Write end shut down"));
		}
		
		void abortRead() override {
			readFrom -> shutdownReadEnd(KJ_EXCEPTION(DISCONNECTED, "Read aborted"));
		}
	};
}

kj::TwoWayPipe newPipe() {
	auto queue1Read = kj::atomicRefcounted<AsyncMessageQueue>();
	auto queue2Read = kj::atomicRefcounted<AsyncMessageQueue>();
	
	auto queue1Write = queue1Read->addRef();
	auto queue2Write = queue2Read->addRef();
	
	kj::TwoWayPipe result;
	result.ends[0] = kj::heap<MessageQueueIOStream>(mv(queue1Read), mv(queue2Write));
	result.ends[1] = kj::heap<MessageQueueIOStream>(mv(queue2Read), mv(queue1Write));
	
	return result;
}	

}