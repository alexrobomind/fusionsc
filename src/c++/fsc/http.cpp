#include <kj/debug.h>

#include "http.h"

namespace fsc {

SimpleHttpServer::SimpleHttpServer(Promise<Own<kj::NetworkAddress>> address, LibraryThread& lt, HttpRoot::Reader data) :
	_headerTable(kj::heap<kj::HttpHeaderTable>()),
	_data(capnp::clone(data)),
	_server(nullptr),
	_port(nullptr),
	_listen(nullptr)
{
	KJ_LOG(WARNING, "Making port info");
	auto portInfo = kj::newPromiseAndFulfiller<unsigned int>();
	//auto listenInfo = kj::newPromiseAndFulfiller<void>();
	
	KJ_LOG(WARNING, "Forking port info");
	_port = portInfo.promise.fork();
	//_listen = listenInfo.promise.fork();
	
	KJ_LOG(WARNING, "Starting server from address");
	_server = address.then([this, portInfo = mv(portInfo)/*, listenInfo = mv(listenInfo)*/, lt = lt->addRef()](Own<kj::NetworkAddress> address) mutable {
		KJ_LOG(WARNING, "Starting listener");
		auto listener = address->listen();
		portInfo.fulfiller->fulfill(listener->getPort());
		
		KJ_LOG(WARNING, "Making server...");
		auto result = std::make_shared<kj::HttpServer>(
			lt -> ioContext().provider -> getTimer(),
			*_headerTable,
			*this
		);
		
		//result->listenHttp(*listener).then([listenInfo = mv(listenInfo)]() mutable { listenInfo.fulfiller->fulfill(); });
		//KJ_LOG(WARNING, "Listening...");
		_listen = result->listenHttp(*listener).attach(mv(listener)).fork();
		
		return result;
	})
	.fork();
}

Promise<void> SimpleHttpServer::drain() {
	return getServer().then([](std::shared_ptr<kj::HttpServer> pSrv) {
		return pSrv -> drain();
	});
}

	
Promise<void> SimpleHttpServer::request(
	kj::HttpMethod method,
	kj::StringPtr url,
	const kj::HttpHeaders& headers,
	kj::AsyncInputStream& requestBody,
	Response& response
) {
	using kj::HttpMethod;
	KJ_LOG(WARNING, "Received request");
	KJ_LOG(WARNING, url);
	
	KJ_REQUIRE(method == HttpMethod::GET, "Can only process method GET");
	
	KJ_LOG(WARNING, "Iterating data");
	for(auto e : _data -> getEntries()) {
		KJ_LOG(WARNING, "Checking entry");
		KJ_LOG(WARNING, e);
		if(e.getLoc() != url)
			continue;
		
		KJ_LOG(WARNING, "Found entry");
		
		KJ_REQUIRE(e.isText());
		auto d = e.getText();
		
		KJ_LOG(WARNING, "Sending");
		auto ostream = response.send(200, "OK", kj::HttpHeaders(*_headerTable)/*, e.getText().size()*/);
		KJ_LOG(WARNING, "Writing");
		auto result = ostream -> write(d.begin(), d.size());
		KJ_LOG(WARNING, "Returning");
		return result.attach(mv(ostream)).then([](){KJ_LOG(WARNING, "Write finished");});
	}
	
	//return kj::NEVER_DONE;
	return response.sendError(404, "Not found", kj::HttpHeaders(*_headerTable));	
}

}