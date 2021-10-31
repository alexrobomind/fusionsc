#include "local.h"

namespace fsc {
	
// === class ThreadHandle ===

ThreadHandle::ThreadHandle(const Library* l) :
	_ioContext(kj::setupAsyncIo()),
	_library(l -> addRef()),
	_executor(kj::getCurrentThreadExecutor())
{}

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
	
	// Prepare a random security token
	h.rng().randomize((ArrayPtr<byte>) sendToken);
	
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
		auto readOp = stream -> read(recvToken.begin(), recvToken.size());
		
		// Check the token and return the stream
		return readOp.then([this, stream2 = mv(stream)]() mutable {
			KJ_REQUIRE((ArrayPtr<byte>) recvToken == (ArrayPtr<byte>) sendToken, "Security token mismatch");
			return mv(stream2);
		});
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
		auto writeOp = stream -> write(sendToken.begin(), sendToken.size());
		
		// After writing the token, return the stream
		return writeOp.then([stream2 = mv(stream)]() mutable {
			return mv(stream2);
		});
	})
	;
}

}