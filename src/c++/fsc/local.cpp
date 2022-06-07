#include <list>

#include "local.h"
#include "data.h"

namespace fsc {
	
// === class LibraryHandle ===

void LibraryHandle::stopSteward() const {
	KJ_LOG(WARNING, "LibraryHandle::stopSteward()");
	storeSteward.stop();
}
	
// === class ThreadHandle ===

ThreadHandle::ThreadHandle(Library l) :
	_ioContext(kj::setupAsyncIo()),
	_library(l -> addRef()),
	_executor(kj::getCurrentThreadExecutor()),
	_dataService(kj::heap<LocalDataService>(l)),
	_filesystem(kj::newDiskFilesystem())
{}

ThreadHandle::~ThreadHandle() {
	//delete _dataService;
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

}