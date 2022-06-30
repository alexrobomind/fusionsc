#include <kj/debug.h>

#include "http.h"

namespace fsc {

SimpleHttpServer::SimpleHttpServer(Promise<Own<kj::NetworkAddress>> address, HttpRoot::Reader data) :
	_headerTable(kj::heap<kj::HttpHeaderTable>()),
	_data(capnp::clone(data)),
	_server(nullptr),
	_port(nullptr),
	_listen(nullptr)
{
	auto portInfo = kj::newPromiseAndFulfiller<unsigned int>();
	_port = portInfo.promise.fork();
	
	_server = address.then([this, portInfo = mv(portInfo)](Own<kj::NetworkAddress> address) mutable {
		auto listener = address->listen();
		portInfo.fulfiller->fulfill(listener->getPort());
		
		auto result = std::make_shared<kj::HttpServer>(
			getActiveThread().ioContext().provider -> getTimer(),
			*_headerTable,
			*this
		);
		
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
	KJ_REQUIRE(method == HttpMethod::GET, "Can only process method GET");
	
	for(auto e : _data -> getEntries()) {
		if(e.getLoc() != url)
			continue;
		
		
		KJ_REQUIRE(e.isText());
		auto d = e.getText();
		auto ostream = response.send(200, "OK", kj::HttpHeaders(*_headerTable), d.size());
		auto result = ostream -> write(d.begin(), d.size());
		return result.attach(mv(ostream));
	}
	
	return response.sendError(404, "Not found", kj::HttpHeaders(*_headerTable));	
}

}