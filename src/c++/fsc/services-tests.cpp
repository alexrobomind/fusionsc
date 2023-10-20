#include <catch2/catch_test_macros.hpp>

#include <kj/array.h>

#include "services.h"
#include "local.h"
#include "magnetics.h"
#include "geometry.h"

using namespace fsc;

TEST_CASE("in-process-server") {
	auto library = newLibrary();
	auto thread = library -> newThread();
	
	auto localFactory = []() {
		Temporary<LocalConfig> config;
		return createLocalResources(config);
	};
	
	auto server = newInProcessServer<LocalResources>(mv(localFactory));
	
	auto connected = server();
	connected.whenResolved().wait(thread -> waitScope());
}

TEST_CASE("http-connect") {
	auto library = newLibrary();
	auto thread = library -> newThread();
	auto& ws = thread -> waitScope();
	auto& ds = thread -> dataService();
	
	// Create a DataRef as service we want to serve
	
	Temporary<LocalConfig> config;
	LocalResources::Client lr = createLocalResources(config);
	
	auto data = kj::heapArray<capnp::byte>(20);
	thread -> rng().randomize(data);
	auto published = ds.publish((capnp::Data::Reader) data);
	
	auto serveRequest = lr.serveRequest();
	serveRequest.setServer(published);
	auto openPort = serveRequest.send().wait(ws).getOpenPort();
	auto port = openPort.getInfoRequest().send().wait(ws).getPort();
	
	auto connReq = lr.connectRequest();
	connReq.setUrl(kj::str("http://localhost:", port));
	auto connection = connReq.send().wait(ws).getConnection();
	
	auto remote = connection.getRemoteRequest().send().wait(ws).getRemote();
	auto copy = ds.download(remote.castAs<DataRef<capnp::Data>>()).wait(ws);
	
	KJ_REQUIRE(copy.get() == data);
	
	remote = nullptr;
	connection = nullptr;
	
	openPort.drainRequest().send().wait(ws);
}

TEST_CASE("ssh-auth", "[.][ssh]") {
	KJ_DBG("Beginning test");
	auto library = newLibrary();
	auto thread = library -> newThread();
	
	Temporary<LocalConfig> config;
	LocalResources::Client localResources = createLocalResources(config.asReader());
	
	auto req = localResources.sshConnectRequest();
	req.setHost("localhost");
	req.setPort(2222);
	
	thread -> timer().afterDelay(1000 * kj::MILLISECONDS).wait(thread -> waitScope());
	
	KJ_DBG("Waiting for connection");
	auto conn = req.send().wait(thread->waitScope()).getConnection();
	
	SECTION("badauth") {
		auto req = conn.authenticatePasswordRequest();
		req.setUser("testuser");
		req.setPassword("wrongPassword");
		REQUIRE_THROWS(req.send().wait(thread -> waitScope()));
	}
	
	SECTION("password") {
		auto req = conn.authenticatePasswordRequest();
		req.setUser("testuser");
		req.setPassword("testpass");
		req.send().wait(thread -> waitScope());
	}
	
	SECTION("pubkey") {
		kj::StringPtr pubKeyData = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCgsTMSpKv2sximRc3vJAOSkeI2B4qYL1E3uUelS7q0SaSA2vqxUDA+ipO1wxZvB82bS4KMX5KXna8q41RNpsK0mwyfE6ftQy8c1loDGzwt6NZuBvlN9zGzbD2tTVpFo+rfHvlhsFalIiJdzsfIF/xoTHHeFyLEweLzhamLp/X7JKrau2pm5TkiUknYYR0MXhCfx714yGdDRCtCNXQvJ1ouXiKA5+XMjpb4GY2+qXdwW3LIug1YzV9cd1FCBi9C1maDSOVBD77Syztgs9kI23xwmB1ImkVbHiorOdgRnsD+s+EU/HnVPCQtidnkE+Tbt5ZkZ4mqWHtaJdmekkBakyF3"_kj;
		
		kj::StringPtr privKeyData =
R"(-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3-CBC,D5A4738309691E79

WTkh0OaxwoYyjn0D9T4+qcAemiBFYekNsX4+aEFvKOX8XfvGzwWlJt3G3oJaF5AS
3uEeQHj8VPPMq4ACL92jU6x4rLklix0zH5fTVEZ3gt61ns+Dnjh4L3K61pqRfjsa
4o/yyC0jOrl8AE5PdBw5jbXZCZPTa8rpHsbAJAODr5LvtZ4c9WXuYXhW1WmklyQE
aPr+LGT5eidIw4E2Tl4Joo1Dn65IkxwLoNe60PytuuZJGva4eoVhaCA4olSPVgsf
AeD7Sg1Uqoq2k3GK8+jo1sFslhrAYxo2Wjcy1SwE2F7+MZTPYKcrvRp6Z47JoWOo
ml36f3mqY17CEtDcTPoZ39J32b1UYRStGFbDMGGqukLGjM1nL3cJ6ozhVaxXsXwt
/pSuo3ycC1lJuGJwb7+xi0X/Sl0lMxgwYhmbTUtdhj3mnUEjVFpalRbkvt1IFQZe
LRpurMzAk2w4BrR/hg7L1r1j3LYHpBGZgjFNZiaW6kt3Gy314su26Gpi+10GQEAp
HXpfFifMLwM+UJkUcATtjN1vcqxbiAaxeCwmYQweIyrmyfsMd9vRHh6Sv/4PnQfd
gHvqCwf5lz6WXosmFqlTIMxAxrHmUGLlqD3l4O8oz9iAAjAGpyj0ovScIDQSNYyJ
4bfMtloyttmr4TItGMAObmSiNNr9qp26s02WbVHYJWt0ZXGkLx1MPofZtz6srFes
Pl/aBoFzVb5aH/Al/DLBosOfUaJ0BJpnrrUhIQ4MS3BWo2b+yl8TEsnYYZ5tq4fP
OnHnrTH1wBSVhRVMxEg0re8CBat04GlH1G9Bc1DbV6UKi3i1Rxhb0A05C+epN1ic
Tm5W39r0ekAFnNbp/skPtte18FJPdCT4EXW6tP7Ex1ExvcojYrLNzt9ZzgeDNAdF
sM8z0MWDo/lTL9vlw5K2N9Y73+SQ37wIW4TBlxktKg05eFZw/3NIfomVYRKXR8Mt
RHv9U7hG1o3lGdkMNPTzSOwUHfeRbDL3sIgZB4gGRhM8HsNtTxVYj8g4SNRtoTRK
qn5KHaz0EpuI6ECbgJt8xM5gVrjK8OTWJiz0x2xpD6fXHiMzhYNy6HaooZ2zvkZy
Y5LOmwkYgCf5Jl2xgJ6NWmrm2RuDcJMpA08iiSucRxennMVJ4TtYeJINkFJ9a4F7
s5utEnJ4rKvJfVCWLMUV7KFGQka+8kxLZRg9/LW97fZAFZ1HmElp/kxPAt2UWsjM
jw0Fyy9fwGymRHmNLjTXMB6BnrtUT3GjjR9Yzl0sWRCIuyE23S+LaqwmVbcrtHHJ
U+03kRUaFAQ2Dvo3l8okSc46HSouy+LqOW/k/gQReono+8ZML8GEoRZ5lWNdL4Qw
Q3xI+fEsQoyTPI5VMN+jzIiUyP6fz7ykYFB6aQntDgSmBaTjSxKnet6PMsN14yoG
aB4siibQxg61FFNxR/eg8d44XZUVrQ91rqch48yHmOkJRWzTOhg6MVAbcONtXBVw
2H5ENc61ZlnKqLGgekIsvtOhmP3wdln1UvkwWvBbt8f/bSEkcm2AAUNscPG8cQGz
TxbJOr1B1BvMKM+VXSQ6eOUrNblFlagl9d7qj+rJUo3OgLYFz47c9w==
-----END RSA PRIVATE KEY-----)"_kj;

		SECTION("keyfile") {
			auto fs = kj::newDiskFilesystem();
			kj::StringPtr pubKeyFilename = "tmp-pubkey";
			kj::StringPtr privKeyFilename = "tmp-privkey";
			
			auto write = [&](kj::StringPtr name, kj::StringPtr data) {
				auto f = fs -> getCurrent().openFile(kj::Path(name), kj::WriteMode::CREATE | kj::WriteMode::MODIFY);
				f -> writeAll(data);
			};
			
			write(pubKeyFilename, pubKeyData);
			write(privKeyFilename, privKeyData);

			kj::StringPtr pass = "testpass"_kj; // Encryption password for private key
			
			auto req = conn.authenticateKeyFileRequest();
			req.setUser("testuser");
			req.setPubKeyFile(pubKeyFilename);
			req.setPrivKeyFile(privKeyFilename);
			req.setKeyPass(pass);
			
			req.send().wait(thread -> waitScope());
		}
	}
}

namespace {
	bool fieldDummyCalled;
	
	class DummyResolver : public FieldResolver::Server {
		using FieldResolver::Server::ResolveFieldContext;
		
		Promise<void> resolveField(ResolveFieldContext ctx) override {
			fieldDummyCalled = true;
			// ctx.setResults(ctx.getParams().getField());
			ctx.getResults().initInvert();
			return READY_NOW;
		}
	};
}