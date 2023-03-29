#include "streams.h"
#include "local.h"

#include <capnp/capability.h>

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
	
	Promise<void> pumpTo(PumpToContext ctx) {
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
	return kj::heap<RIImpl>(*this, mv(backend));
}

RemoteOutputStream::Client StreamConverterImpl::toRemote(Own<AsyncOutputStream> backend) {
	return kj::heap<ROImpl>(*this, mv(backend));
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

}