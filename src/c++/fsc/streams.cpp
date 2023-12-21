#include "streams.h"
#include "local.h"

#include <capnp/capability.h>
#include <kj/async-queue.h>

using kj::AsyncOutputStream;
using kj::AsyncInputStream;
using kj::AsyncIoStream;

namespace fsc {
	
namespace {

struct StreamConverterImpl : public StreamConverter, public kj::Refcounted {
	RemoteInputStream::Client toRemote(Own<kj::AsyncInputStream>) override;
	RemoteOutputStream::Client toRemote(Own<kj::AsyncOutputStream>) override;
	
	Promise<Own<kj::AsyncInputStream>> fromRemote(RemoteInputStream::Client clt) override;
	Promise<Own<kj::AsyncOutputStream>> fromRemote(RemoteOutputStream::Client clt) override;
	
	Own<StreamConverterImpl> addRef() { return kj::addRef(*this); }
	
private:
	capnp::CapabilityServerSet<RemoteOutputStream> wrappedOStreams;
	capnp::CapabilityServerSet<RemoteInputStream> wrappedIStreams;
};

struct ROImpl : public RemoteOutputStream::Server {
	Maybe<Own<AsyncOutputStream>> backend;
	
	ROImpl(StreamConverterImpl& conv, Own<AsyncOutputStream> backend) :
		backend(mv(backend))
	{
		(void) conv;
	}	
	
	Promise<void> write(WriteContext ctx) {
		FSC_REQUIRE_MAYBE(pBackend, backend, "Stream already closed");
		auto data = ctx.getParams().getData();
		return (*pBackend) -> write(data.begin(), data.size());
	}
	
	Promise<void> eof(EofContext ctx) {
		backend = nullptr;
		return READY_NOW;
	}
	
	Promise<void> flush(FlushContext ctx) {
		return READY_NOW;
	}
};

struct RIImpl : public RemoteInputStream::Server {
	Own<StreamConverterImpl> converter;
	Own<AsyncInputStream> backend;
	bool locked = false;
	
	RIImpl(StreamConverterImpl& conv, Own<AsyncInputStream> backend) :
		converter(conv.addRef()),
		backend(mv(backend))
	{}
	
	Promise<void> pumpTo(PumpToContext ctx) override {
		auto ros = ctx.getParams().getTarget();
		
		KJ_REQUIRE(!locked, "Stream already in use");
		locked = true;
		
		return converter -> fromRemote(mv(ros))
		.then([this](Own<AsyncOutputStream> target) mutable {
			return backend -> pumpTo(*target).attach(mv(target));
		}).then([this, ctx](size_t pumpedBytes) mutable {
			ctx.initResults().setPumpedBytes(pumpedBytes);
			locked = false;
		}, [this](kj::Exception&& e) mutable {
			locked = false;
			kj::throwFatalException(mv(e));
		});
	}
	
	Promise<void> readAllBinary(ReadAllBinaryContext ctx) override {
		locked = true;
		return backend -> readAllBytes()
		.then([ctx](Array<kj::byte> bytes) mutable {
			ctx.initResults().setData(bytes);
		});
	}
	
	Promise<void> readAllString(ReadAllStringContext ctx) override {
		locked = true;
		return backend -> readAllText()
		.then([ctx](kj::String str) mutable {
			ctx.initResults().setText(str);
		});
	}
};

struct RemotelyBackedOutputStream : public kj::AsyncOutputStream {
	RemoteOutputStream::Client backend;
	
	RemotelyBackedOutputStream(RemoteOutputStream::Client clt) :
		backend(mv(clt))
	{}
	
	~RemotelyBackedOutputStream() {
		getActiveThread().detach(backend.eofRequest().send().ignoreResult());
	}
	
	Promise<void> write(const void* buffer, size_t size) override {
		kj::ArrayPtr<const kj::byte> dataPtr((const kj::byte*) buffer, size);
		
		auto req = backend.writeRequest();
		req.setData(dataPtr);
		return req.send();
	}
	
	Promise<void> write(ArrayPtr<const ArrayPtr<const kj::byte>> pieces) override {
		Promise<void> result = READY_NOW;
		
		for(auto piece : pieces) {
			result = result.then([this, piece]() mutable {
				write(piece.begin(), piece.size());
			});
		}
		
		return result;
	}
	
	Promise<void> whenWriteDisconnected() override { return NEVER_DONE; }
};

struct RemotelyBackedInputStream : public kj::AsyncInputStream {
	Own<AsyncInputStream> pipeEnd;
	Promise<void> pumpTask = nullptr;
	kj::Canceler canceler;
	
	RemotelyBackedInputStream(RemoteInputStream::Client clt) {
		auto pipe = kj::newOneWayPipe();
		
		// Take ownership of the read end
		pipeEnd = mv(pipe.in);
		
		// Start the pump on the remote end
		auto remoteEnd = getActiveThread().streamConverter().toRemote(mv(pipe.out));
		auto req = clt.pumpToRequest();
		req.setTarget(mv(remoteEnd));
		pumpTask = req.send().ignoreResult().eagerlyEvaluate([this](kj::Exception e) {
			canceler.cancel(e);
		});
	}

	Promise<size_t> tryRead(void* buf, size_t min, size_t max) override {
		return canceler.wrap(pipeEnd -> tryRead(buf, min, max));
	}
};

}

// === class StreamConverter ===

StreamConverter::~StreamConverter() noexcept(false) {}

// === class StreamConverterImpl ===

RemoteInputStream::Client StreamConverterImpl::toRemote(Own<AsyncInputStream> backend) {
	return wrappedIStreams.add(kj::heap<RIImpl>(*this, mv(backend)));
}

RemoteOutputStream::Client StreamConverterImpl::toRemote(Own<AsyncOutputStream> backend) {
	return wrappedOStreams.add(kj::heap<ROImpl>(*this, mv(backend)));
}

Promise<Own<AsyncOutputStream>> StreamConverterImpl::fromRemote(RemoteOutputStream::Client backend) {
	return wrappedOStreams.getLocalServer(backend)
	.then([backend](Maybe<RemoteOutputStream::Server&> unwrapped) mutable -> Own<AsyncOutputStream> {
		KJ_IF_MAYBE(pUnwrapped, unwrapped) {
			ROImpl& impl = static_cast<ROImpl&>(*pUnwrapped);

			FSC_REQUIRE_MAYBE(pStream, impl.backend, "Stream already in use or closed");
			Own<AsyncOutputStream> oStream = mv(*pStream);
			impl.backend = nullptr;
			return mv(oStream);
		}
		
		return kj::heap<RemotelyBackedOutputStream>(backend);
	}).attach(addRef());
}

Promise<Own<AsyncInputStream>> StreamConverterImpl::fromRemote(RemoteInputStream::Client backend) {
	return wrappedIStreams.getLocalServer(backend)
	.then([this, backend](Maybe<RemoteInputStream::Server&> unwrapped) mutable -> Own<AsyncInputStream> {
		KJ_IF_MAYBE(pUnwrapped, unwrapped) {
			RIImpl& impl = static_cast<RIImpl&>(*pUnwrapped);
			
			KJ_REQUIRE(!impl.locked, "Stream already in use");
			impl.locked = true;
			
			return mv(impl.backend);
		}
		
		return kj::heap<RemotelyBackedInputStream>(backend);
	}).attach(addRef());
}

Own<StreamConverter> newStreamConverter() {
	return kj::refcounted<StreamConverterImpl>();
}

// ================================= Buffered input streams ==================================

namespace {
	
constexpr size_t BLOCK_SIZE = 1024;

struct InputBuffer : public kj::Refcounted {
	using L = std::list<kj::Array<kj::byte>>;
	
	struct Cursor {
		L::iterator pos;
		size_t bytes = 0;
		
		kj::ListLink<Cursor> link;
		
		Cursor() = default;
		Cursor(Cursor&&) = delete; // ListLink can not be moved
		Cursor(const Cursor& other) :
			pos(other.pos), bytes(other.bytes)
		{}
	};
	
	InputBuffer(Own<kj::AsyncInputStream>&& is, uint64_t pLimit) :
		sizeLimit(pLimit),
		inputStream(mv(is))
	{
		newBlock();
		startPump();
	}
	
	Promise<void> pumpTask() {
		auto buf = inputBuffer();
		if(buf.size() == 0) {
			// Buffer is full, check if we can make a new one
			if(size + BLOCK_SIZE <= sizeLimit) {
				// Yes, so let's make one
				newBlock();
				buf = inputBuffer();
			} else {
				// Naah, max size exceeded
				// We have to block (for now)
				pumpActive = false;
				return READY_NOW;
			}
		}
		
		return inputStream -> tryRead(buf.begin(), 1, buf.size())
		.then([this](size_t bytesRead) -> Promise<void> {
			advance(bytesRead);
			
			if(bytesRead == 0) {
				closed = true;
				pumpActive = false;
				return READY_NOW;
			}
			return pumpTask();
		});
	}
	
	void startPump() {
		if(pumpActive)
			return;
		
		pumpActive = true;
		pump = pumpTask().eagerlyEvaluate([this](kj::Exception&& e) {
			advance(0);
			closed = true;
		});
	}
	
	size_t consume(kj::ArrayPtr<kj::byte> output, Cursor& c) {		
		size_t consumed = 0;
		
		while(true) {
			if(consumed == output.size())
				break;
			
			// Check if we can consume from the current element
			size_t bufAvail = (c.pos == writePosition.pos) ? writePosition.bytes : c.pos -> size();
			size_t inRemaining = bufAvail - c.bytes;
			
			// If not, check if we are at end or advance
			if(inRemaining == 0) {
				if(c.pos == writePosition.pos)
					break; // Nothing left to copy
				
				++c.pos;
				c.bytes = 0;
				
				gc();
				
				continue;
			}
			
			// OK, we have something to copy
			size_t copyOver = kj::min(inRemaining, output.size() - consumed);
			memcpy(output.begin() + consumed, c.pos -> begin() + c.bytes, copyOver);
			
			c.bytes += copyOver;
			consumed += copyOver;
		}
		
		return consumed;
	}
	
	void gc() {
		bool freed = false;
		
		while(true) {
			// Perhaps we can now free the first block of the buffer
			bool canFree = true;
			for(auto& c : readers) {
				if(c.pos == data.begin())
					canFree = false;
			}
			
			if(canFree) {
				data.pop_front();
				freed = true;
			} else {
				break;
			}
		}
		
		if(freed) {			
			// The pump might have blocked due to running out of space
			// Try to restart it
			startPump();
		}
	}
		
	kj::ArrayPtr<kj::byte> inputBuffer() {
		return writePosition.pos -> slice(writePosition.bytes, writePosition.pos -> size());
	}
	
	void advance(size_t bytes) {
		writePosition.bytes += bytes;
		
		while(!queue.empty())
			queue.fulfill(cp(bytes));
	}
	
	Promise<size_t> awaitRead() {
		if(!closed)
			return queue.wait();
		
		return (size_t) 0;
	}
	
	void reset(Cursor& c) {
		c.pos = data.begin();
		c.bytes = 0;
	}
	
	const uint64_t sizeLimit;
	kj::List<Cursor, &Cursor::link> readers;

private:
	void newBlock() {
		data.emplace_back(kj::heapArray<kj::byte>(BLOCK_SIZE));
		
		writePosition.pos = --data.end();
		writePosition.bytes = 0;
	}
	
	kj::WaiterQueue<size_t> queue;
	
	L data;
	Cursor writePosition;
	
	size_t size = 0;
	
	Own<kj::AsyncInputStream> inputStream;
	Maybe<Promise<void>> pump;
	
	bool closed = false;
	bool pumpActive = false;
};

struct BufferedInputStream : public kj::AsyncInputStream {
	Own<InputBuffer> buf;
	InputBuffer::Cursor cursor;
	
	BufferedInputStream(InputBuffer& buf) :
		buf(kj::addRef(buf))
	{
		buf.reset(cursor);
		buf.readers.add(cursor);
	}
	
	BufferedInputStream(InputBuffer& buf, InputBuffer::Cursor& c) :
		buf(kj::addRef(buf)),
		cursor(c)
	{
		buf.readers.add(cursor);
	}
	
	~BufferedInputStream() {
		buf -> readers.remove(cursor);
		
		if(!buf -> readers.empty())
			buf -> gc();
	}
	
	Promise<size_t> tryRead(void* bufPtr, size_t minBytes, size_t maxBytes) override {
		ArrayPtr<byte> outBuf((byte*) bufPtr, minBytes);
		
		// Consume as many bytes as possible
		size_t initiallyConsumed = buf -> consume(outBuf, cursor);
		
		if(initiallyConsumed >= minBytes)
			return initiallyConsumed;
		
		return waitUntilAvailable(minBytes - initiallyConsumed)
		.then([this, outBuf, initiallyConsumed](size_t extra) mutable {
			// Consume as much as possible again
			size_t extraConsumed = buf -> consume(outBuf.slice(initiallyConsumed, outBuf.size()), cursor);
			return initiallyConsumed + extraConsumed;
		});
	}
	
	Maybe<Own<kj::AsyncInputStream>> tryTee(uint64_t limit) override {
		// If the limit is smaller than the limits encoded in this buffer,
		// we need a more memory-conservative tee
		if(limit < buf -> sizeLimit)
			return nullptr;
		
		return kj::heap<BufferedInputStream>(*buf, cursor);
	}
	
	Promise<size_t> waitUntilAvailable(size_t required, size_t fulfilled = 0) {
		if(fulfilled >= required)
			return fulfilled;
		
		return buf -> awaitRead()
		.then([this, required, fulfilled](size_t next) -> Promise<size_t> {
			if(next == 0)
				return fulfilled;
			
			return waitUntilAvailable(required, fulfilled + next);
		});
	}
};

}

Own<kj::AsyncInputStream> buffer(Own<kj::AsyncInputStream>&& is, uint64_t limit) {
	auto buf = kj::refcounted<InputBuffer>(mv(is), limit);
	return kj::heap<BufferedInputStream>(*buf);
}

// ======================================== Output multiplexer ====================================

namespace {

struct OutputMultiplexerImpl : public MultiplexedOutputStream, kj::Refcounted {
	kj::WaiterQueue<int> queue;
	
	Own<kj::AsyncOutputStream> backend;
	bool blocked = false;
	
	ForkedPromise<void> wwd;
	
	OutputMultiplexerImpl(Own<kj::AsyncOutputStream>&& nb) :
		backend(mv(nb)),
		wwd(backend -> whenWriteDisconnected().fork())
	{}
	
	~OutputMultiplexerImpl() {}
	
	Own<MultiplexedOutputStream> addRef() override {
		return kj::addRef(*this);
	}
	
	template<typename T, typename Func>
	Promise<T> blocking(Func&& f) {
		if(blocked) {
			return queue.wait()
			.then([this, f2 = fwd<Func>(f)](int ignoreThis) mutable {
				(void) ignoreThis;
				return blocking<T>(fwd<Func>(f2));
			});
		}
		
		blocked = true;
		return kj::evalNow(fwd<Func>(f))
		.attach(kj::defer([self = kj::addRef(*this)]() mutable { self -> unblock(); }));
	}
	
	void unblock() {
		blocked = false;
		
		if(!queue.empty())
			queue.fulfill(0);
	}
	
	Promise<void> write(const void* buffer, size_t size) override {
		return blocking<void>([this, buffer, size]() {
			return backend -> write(buffer, size);
		});
	}
	
	Promise<void> write(ArrayPtr<const ArrayPtr<const byte>> pieces) override {
		return blocking<void>([this, pieces]() {
			return backend -> write(pieces);
		});
	}
	
	Promise<void> whenWriteDisconnected() override {
		return wwd.addBranch();
	}
};

}

Own<MultiplexedOutputStream> multiplex(Own<kj::AsyncOutputStream>&& os) {
	return kj::refcounted<OutputMultiplexerImpl>(mv(os));
}

// ===================== std::stream compatibility layer =====================


namespace {
	struct OStreamBuffer : public virtual std::streambuf {		
		kj::BufferedOutputStream& os;
		
		OStreamBuffer(kj::BufferedOutputStream& nos) :
			os(nos)
		{
			resetBuffer();
		}
		
		void resetBuffer() {
			auto buf = os.getWriteBuffer();
			setp((char*) buf.begin(), (char*) buf.end());
		}
				
		int overflow(int c = EOF) override {
			// Write buffered data
			os.write(pbase(), (pptr() - pbase()));
			
			resetBuffer();
			
			if(c != EOF) {
				*pptr() = c;
				pbump(1);
			}
			
			return c;
		}
		
		std::streamsize xsputn(const char* s, std::streamsize n) override {
			os.write(pbase(), (pptr() - pbase()));
			os.write(s, n);
			
			resetBuffer();
			
			return n;
		}
	};
	
	struct IStreamBuffer : public virtual std::streambuf {
		kj::BufferedInputStream& is;
		
		IStreamBuffer(kj::BufferedInputStream& nis) :
			is(nis)
		{}
		
		void syncStream() {
			is.skip(gptr() - eback());
		}
		
		kj::ArrayPtr<const kj::byte> syncBuf() {	
			auto buf = is.tryGetReadBuffer();
			setg((char*) buf.begin(), (char*) buf.begin(), (char*) buf.end());
			
			return buf;
		}
		
		std::streamsize xsgetn(char* s, std::streamsize n) override {
			syncStream();
			auto result = is.tryRead(s, n, n);
			syncBuf();
			return result;
		}
		
		int underflow() override {
			syncStream();
			auto buf = syncBuf();
			
			if(buf.size() == 0)
				return EOF;
			
			return buf[0];
		}
	};
}
	
Own<std::istream> asStdStream(kj::BufferedInputStream& is) {
	auto buf = kj::heap<IStreamBuffer>(is);
	auto stream = kj::heap<std::istream>(buf.get());
	
	return stream.attach(mv(buf));
}

Own<std::ostream> asStdStream(kj::BufferedOutputStream& is) {
	auto buf = kj::heap<OStreamBuffer>(is);
	auto stream = kj::heap<std::ostream>(buf.get());
	
	return stream.attach(mv(buf));
}

}