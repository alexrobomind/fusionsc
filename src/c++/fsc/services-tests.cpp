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
	KJ_DBG("Server created. Connecting ...");
	
	auto connected = server();
	KJ_DBG("Connection formed, waiting for resolution ...");
	connected.whenResolved().wait(thread -> waitScope());
	KJ_DBG("Resolved");
}

TEST_CASE("http-connect") {
	auto library = newLibrary();
	auto thread = library -> newThread();
	auto& ws = thread -> waitScope();
	auto& ds = thread -> dataService();
	
	// Create a DataRef as service we want to serve
	
	Temporary<LocalConfig> config;
	auto lr = createLocalResources(config);
	
	auto data = kj::heapArray<capnp::byte>(20);
	thread -> rng().randomize(data);
	auto published = ds.publish((capnp::Data::Reader) data);
	
	auto serveRequest = lr.serveRequest();
	serveRequest.setServer(published);
	auto openPort = serveRequest.send().wait(ws).getOpenPort();
	auto port = openPort.getInfoRequest().send().wait(ws).getPort();
	KJ_DBG(port);
	
	auto connReq = lr.connectRequest();
	connReq.setUrl(kj::str("http://localhost:", port));
	auto connection = connReq.send().wait(ws).getConnection();
	
	auto remote = connection.getRemoteRequest().send().wait(ws).getRemote();
	auto copy = ds.download(remote.castAs<DataRef<capnp::Data>>()).wait(ws);
	
	KJ_REQUIRE(copy.get() == data);
	
	remote = nullptr;
	connection = nullptr;
	
	openPort.drainRequest().send().wait(ws);
	
	KJ_DBG("Finished test");
}

TEST_CASE("ssh-auth", "[.][ssh]") {
	KJ_DBG("Beginning test");
	auto library = newLibrary();
	auto thread = library -> newThread();
	
	Temporary<LocalConfig> config;
	auto localResources = createLocalResources(config.asReader());
	
	auto req = localResources.sshConnectRequest();
	req.setHost("localhost");
	req.setPort(2222);
	
	thread -> timer().afterDelay(1000 * kj::MILLISECONDS).wait(thread -> waitScope());
	
	KJ_DBG("Waiting for connection");
	auto conn = req.send().wait(thread->waitScope()).getConnection();
	
	SECTION("badauth") {
		KJ_DBG("Testing wrong password");
		auto req = conn.authenticatePasswordRequest();
		req.setUser("testuser");
		req.setPassword("wrongPassword");
		REQUIRE_THROWS(req.send().wait(thread -> waitScope()));
		KJ_DBG("OK");
	}
	
	SECTION("password") {
		KJ_DBG("Testing right password");
		auto req = conn.authenticatePasswordRequest();
		req.setUser("testuser");
		req.setPassword("testpass");
		req.send().wait(thread -> waitScope());
		KJ_DBG("OK");
	}
	
	SECTION("keyfile") {
		KJ_DBG("Testing pubkey");
		kj::StringPtr pubKeyFile = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCKxUZ8uB2rKNM1Eb1HMUntnkQkLGPgJfi7iO1/JfQsvUrK5Il8PE0ZIj7JnZNjPKZjZ55tbmNW9/GX5XF26xLTAWnzTVEt0nipStZCjzmvdT/hffqSAaP0rleWGVf6oG/BebKvn1axvkFuoCVKAa9LxdvDIJ3cUr2bWFVwSqo3zp14+HCBZ7Wmi9CicRuHbuR0vBL1MopLPAJXXsJ/en3N6253+F9m2cSYy62ziCPRHth1xpz19TU4f4oIjQkqzFpDN7dFHco6APu11nILz6jpeeltSWMSaHa75Qch5MullXGn7wJ3pqF5EJgl6zEW3WRFtT2P3THcMAlfmc9XPAyp"_kj;
		
		kj::StringPtr privKeyFile =
R"(-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3-CBC,4A902C91FB0E52F7

mM9IzzI/9k7UHWqTDOsMGV/4By9O2wQ+InCvePo0gi75FMV+AKRNO8itnXqyLK7k
Espc0O52TM9z09mUkdMBKFki47cN0ZA3pn75ROtZqc0QxdczPnSsZ2aFbOyLaw0Q
+//jiRC1RLNcygvLDA1sLKN61PtYPmgT+wR3GxF7lIVYJIcS0kxh0y2cXwf29fBp
JwkkGmWbxyIFDzSlsHbgdwI2PTnhmFgpIos1lhCKYfwuSP4wHGIH+uFBLuZrho0K
U9l9EYJYd5QiRQXN2v4sQjgVSTN4jHrBJaiuwDjswecEPIBD/BMRN43mjpLRGGOJ
fBd6z40lR6Of9ws/JnS6pA3IJihlfBGc05cULw2Ythk5SqokXz3JCiGuOLDKcE+E
26bL38osdBUwxaeOBfAYEAuX2BFyeGPaIbh3ursQoylILaAdM/SD6zxGl8wyEYQy
LnJgFP8HoJHwFcytb1j7y9yQREwYbtfGShbMrugE1NjNmWXLT3r6z5bAIrumMICj
r9c6QfmFWDtMDXEvew9NNb6WwxZKINjLM7lrUagtefhU1/nRYybBWZgDOGKET0/V
pmqpOAjufjkV9isUTOOJMUcPMpNWmMIH97QRJBd1+e/tmcGzCAXobAUbpRY33Mf3
7gk93Xcbus0nzxDc7PednjMvEcmqtGf9Rj1pr5l1gKDC9nN8PM0A/ojiCfIySkN/
qrbJMr0fbzhqGoa9ihPse7TxqLPo7yPRzgMyTinJ87Meqaw+xLyja+DDDMrXonmi
O8IPhwnFNIEpVTzJ5+jO57pAoR0KIBFyFiCExMD+KqfWrLAC2L7EfX5vs1l9sbXa
6hEVyqYj2SzOTxaBvRssNZJCOSo+Qn62smXrmz4ke3yAq6VNcn4zNaokarrug8Bd
C8YhfM/gzQ6XkK8cwTYQw9j+l/m6gyBO0Jv/9pkLtFlkrLCFL7e1l0+rrz/X7ieU
bB+CkLIe3m6fQJLp9yYLa5q+wzNotmc/pkvBX+WpE2+19H3MbIcImPMpjrC8h9qY
A7/s3h9O0G5NKRKsxlgK33sApJxcj/1L3PYwTNm29/sWniQd3wZf7Tvt0zsdPUoa
O2/BueK3iCUROuh18WmPwB+7KkK4eloYCDYaQKQRPAI6Pv09Ufr0ITK1uBIElbuC
wnKq/dLGiGAtmpIy+tHxmFeoT7OHDMRwxnMGkVznWI9LzJHVWAX9/r+BFvFqvLxF
RW4wvK+QXtoBHtJ/zK2IfaTxeKZJb+Mo73jUWTLyCjna2mArkcjdUG/CpC1S/FEJ
SIxAXdt4xZ8bmaFXCyC0+ecbDK5JtEe2h/kASPiJW0AsbHEDi+mejwixSUpE1fR7
Y+EQMkIWpe5HQwkLgPdvhGG7qs8EVB3k8AK+Qg0bg0qLM7BTMSdaaIYd+YKZ46N4
8pY3ts5k0hWXgD7Wo6DFYCDmQuoN4NqwwQQnslP6AGQEjdQDNfXH1LGVk7Stfn/5
FVcDKmQJR8kkCclx0dh69i9GxQTXcnCTqWB5h0ZZxB6H3zYIDrqiTqrjpRacNV0m
bHOU0EeCojcqr8tTQj4LybxTuTXDi+oBX0HSmoHC3YYq9egDSeOj2g==
-----END RSA PRIVATE KEY-----
)"_kj;

		kj::StringPtr pass = "testpass"_kj; // Encryption password for private key
		
		auto req = conn.authenticateKeyDataRequest();
		req.setUser("testuser");
		req.setPubKey(pubKeyFile);
		req.setPrivKey(privKeyFile);
		req.setKeyPass(pass);
		
		req.send().wait(thread -> waitScope());
		KJ_DBG("OK");
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