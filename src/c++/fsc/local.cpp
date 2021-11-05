#include "local.h"

namespace fsc {
	
// === class ThreadHandle ===

ThreadHandle::ThreadHandle(const LibraryHandle* l) :
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
	
	// Prepare a random security tokens
	h.rng().randomize((ArrayPtr<byte>) sendToken1);
	h.rng().randomize((ArrayPtr<byte>) sendToken2);
	
	KJ_LOG(WARNING, "Randomized");
	
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
		KJ_LOG(WARNING, (ArrayPtr<byte>) sendToken1);
		KJ_LOG(WARNING, (ArrayPtr<byte>) recvToken1);
		auto readOp = stream -> read(recvToken1.begin(), recvToken1.size());
		KJ_LOG(WARNING, "readOp1");
		
		auto checkFun = [this] () {
			KJ_LOG(WARNING, "checkFun1");
			KJ_LOG(WARNING, (ArrayPtr<byte>) sendToken1);
			KJ_LOG(WARNING, (ArrayPtr<byte>) recvToken1);
			KJ_REQUIRE((ArrayPtr<byte>) recvToken1 == (ArrayPtr<byte>) sendToken1, "Security token mismatch");
			KJ_LOG(WARNING, "checkFun1 OK");
		};
		KJ_LOG(WARNING, "checkFun1 created");
		checkFun();
		
		/*auto writeFun = [this, &stream = *stream] () {
			KJ_LOG(WARNING, "writeFun1");
			return stream.write(sendToken2.begin(), sendToken2.size());
		};*/
		
		auto returnStream = [stream2 = mv(stream)] () mutable {
			KJ_LOG(WARNING, "returnStream1");
			return mv(stream2);
		};
		
		return readOp.then(checkFun)/*.then(writeFun)*/.then(returnStream);
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
		KJ_LOG(WARNING, (ArrayPtr<byte>) sendToken1);
		KJ_LOG(WARNING, (ArrayPtr<byte>) recvToken1);
		auto writeOp = stream -> write(sendToken1.begin(), sendToken1.size());
		KJ_LOG(WARNING, "writeOp2");
		
		/*auto readFun = [this, &stream = *stream] () {
			KJ_LOG(WARNING, "readFun2");
			return stream.read(recvToken2.begin(), recvToken2.size());
		};
		
		auto checkFun = [this] () {
			KJ_LOG(WARNING, "checkFun2");
			KJ_REQUIRE((ArrayPtr<byte>) recvToken2 == (ArrayPtr<byte>) sendToken2, "Security token mismatch");
			KJ_LOG(WARNING, "checkFun2 OK");
		};*/
		
		auto returnStream = [stream2 = mv(stream)] () mutable {
			KJ_LOG(WARNING, "returnStream2");
			return mv(stream2);
		};
		
		return writeOp/*.then(readFun).then(checkFun)*/.then(returnStream);
	})
	;
}

}