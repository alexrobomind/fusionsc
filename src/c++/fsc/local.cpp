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

ThreadHandle::ThreadHandle(Library l) :
	_ioContext(kj::setupAsyncIo()),
	_library(l -> addRef()),
	_executor(kj::getCurrentThreadExecutor()),
	_dataService(kj::heap<LocalDataService>(l)),
	_filesystem(kj::newDiskFilesystem())
{
	KJ_REQUIRE(current == nullptr, "Can only have one active ThreadHandle / LibraryThread per thread");
	current = this;
}

ThreadHandle::~ThreadHandle() {
	KJ_REQUIRE(current == this, "Destroying LibraryThread in wrong thread") {}
	current = nullptr;
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
			Promise<void> task = kj::evalNow(mv(func));
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

// ======================= crossThreadPipe ================================

namespace {
	struct AsyncMessageQueue : kj::AtomicRefcounted {
		struct DstBufferQueueEntry {
			kj::ListLink<DstBufferQueueEntry> link;
			kj::ArrayPtr<byte> ptr;
			Own<PromiseFulfiller> whenConsumed;
			size_t minBytes;
		};
		
		struct SrcBufferQueueEntry {
			kj::ListLink<SrcBufferQueueEntry> link;
			kj::ArrayPtr<const byte> ptr;
			Maybe<Own<CrossThreadPromiseFulfiller>> whenConsumed;
		};
		
		struct Shared {
			kj::List<SrcBufferQueueEntry, &SrcBufferQueueEntry::link> queue;
			Own<CrossThreadPromiseFulfiller> newInputsReady;
			
			Maybe<kj::Exception> sendError;
		};
		
		struct ReaderPrivate {
			kj::List<SrcBufferQueueEntry, &SrcBufferQueueEntry::link> inQueue;
			kj::List<DstBufferQueueEntry, &DstBufferQueueEntry::link> outQueue;

			size_t inConsumed;
			size_t outConsumed;
			
			Promise<void> newInputsReady;
					
			Maybe<Promise<void>> pumpLoop;
			Maybe<kj::Exception> readError;
		};
		
		kj::MutexGuarded<Shared> shared;
		ReaderPrivate readerPrivate;
		Canceler readCanceler;
		
		Own<const AsyncMessageQueue> addRef() const { return kj::atomicAddRef(*this); }
		
		Promise<void> send(kj::ArrayPtr<kj::ArrayPtr<const byte>> queue) const {
			auto locked = shared.lockExclusive();
			
			KJ_IF_MAYBE(locked->sendError) {
				return locked->sendError;
			}
			
			for(auto ptr : queue) {
				BufferQueueEntry* e = new BufferQueueEntry();
				e.ptr = ptr;
				locked.queue.add(*e);
			}
			
			auto paf = kj::newPromiseAndCrossThreadFulfiller();
			BufferQueueEntry* barrier = new BufferQueueEntry();
			barrier.whenConsumed = mv(paf.fulfiller);
			locked.queue.add(*e);
		
			if(locked.newInputsReady().get() != nullptr)
				locked.newInputsReady->fulfill();
		}
		
		void shutdownWriteEnd(kj::Exception e) const {
			auto locked = shared.lockExclusive();
			
			locked->sendError = e;
			locked->newInputsReady->reject(e);
			
			auto& queue = shared->queue;
			
			for(auto& e : queue) {
				queue.remove(e);
					
				KJ_IF_MAYBE(pWC, eIn.whenConsumed) {
					(**pWC).reject(e);
				}
				
				delete &eIn;
			}
		}
		
		Promise<void> read(kj::ArrayPtr<byte> out, size_t minBytes) {
			auto& mine = readerPrivate;
			
			// If the read end has shut down, return reason
			KJ_IF_MAYBE(pErr, mine.readError) {
				return *pErr;
			}
			
			KJ_IF_MAYBE(dontCare, mine.pumpLoop) {
			} else {
				mine.pumpLoop = readCanceler.wrap(pump()).eagerlyEvaluate([this, &mine](kj::Exception e){
					// Pump the local queue one last time
					pumpLocal();
					
					mine.readError = e;
					
					// Cancel all outstanding read requests
					auto& queue = mine.outQueue;
					for(auto& eOut : queue) {
						queue.remove(eOut);
						eOut.whenConsumed.reject(e);
						delete &eOut;
					}
				});
			}
			
			auto paf = kj::newPromiseAndFulfiller();
			
			auto e = new BufferQueueEntry();
			e -> ptr = out;
			e -> minBytes = minBytes;
			e -> whenConsumed = mv(paf.fulfiller);
			
			mine.outQueue.add(*e);
			
			pumpLocal();
			
			return mv(paf.promise);
		}
		
		void shutdownReadEnd(kj::Exception&& reason) {
			readCanceler.cancel(mv(reason));
		}
		
	private:
		void pumpLocal() {
			auto& mine = readerPrivate;
			
			while(true) {
				if(inQueue.size() == 0 || outQueue.size() == 0)
					break;
				
				auto& eIn = *inQueue.begin();
				auto& eOut = *outQueue.begin();
				
				auto inBuf = eIn.ptr;
				auto outBuf = eOut.ptr;
				
				size_t inRem = inBuf.size() - mine.inConsumed;
				size_t outRem = outBuf.size() - mine.outConsumed;
				
				size_t rem = std::min(inRem, outRem);
				
				memcpy(outBuf.begin() + mine.outConsumed, inBuf.begin() + mine.inConsumed, rem);
				
				mine.inConsumed += rem;
				mine.outConsumed += rem;
				
				if(mine.inConsumed >= inBuf.size()) {
					inQueue.remove(eIn);
					
					KJ_IF_MAYBE(pWC, eIn.whenConsumed) {
						(**pWC).fulfill();
					}
					
					delete &eIn;
					mine.inConsumed = 0;
				}
				
				if(mine.outConsumed >= outBuf.size()) {
					outQueue.remove(eOut);
					eOut.whenConsumed -> fulfill();
					
					delete &eOut;
					mine.outConsumed = 0;
				}
			}
			
			while(outQueue.size() > 0) {
				auto& eOut = *outQueue.begin();
				
				if(mine.outConsumed >= eOut.minBytes) {
					outQueue.remove(eOut);
					eOut.whenConsumed -> fulfill();
					
					delete &eOut;
					mine.outConsumed = 0;
				} else {
					break;
				}
			}
		}
		
		void steal() {
			auto locked = shared.lockExclusive();
			auto& mine = readerPrivate;
			
			for(auto& e : locked -> queue) {
				locked -> queue.remove(e);
				mine.inQueue.add(e);
			}
			
			KJ_IF_MAYBE(pErr, locked->sendError) {
				mine.newInputsReady = *pErr:
			} else {
				auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
				locked -> newInputsReady = mv(paf.fulfiller);
				mine.newInputsReady = mv(paf.promise);
			}
		}
		
		Promise<void> pump() {
			steal();
			pumpLocal();
			
			return mine.newInputsReady.then(KJ_BIND_METHOD(*this, pump));
		}
	};
	
}

}