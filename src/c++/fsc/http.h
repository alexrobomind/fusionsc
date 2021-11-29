#pragma once

#include <kj/compat/http.h>

#include <fsc/http.capnp.h>

#include <memory>

#include "common.h"
#include "local.h"

namespace fsc {

struct SimpleHttpServer : public kj::HttpService {
	SimpleHttpServer(Promise<Own<kj::NetworkAddress>> address, LibraryThread& lt, HttpRoot::Reader data);
	
	inline Promise<unsigned int> getPort() { return _port.addBranch(); }
	Promise<void> drain();
	
	inline Promise<std::shared_ptr<kj::HttpServer>> getServer() { return _server.addBranch(); }
	
	Promise<void> request(
		kj::HttpMethod method,
		kj::StringPtr url,
		const kj::HttpHeaders& headers,
		kj::AsyncInputStream& requestBody,
		Response& response
	);
	
	//inline Promise<void> listen() { return _listen.addBranch(); }
	inline Promise<void> listen() { return getServer().then([this](std::shared_ptr<kj::HttpServer> s) { return _listen.addBranch(); }); }

private:
	Own<kj::HttpHeaderTable> _headerTable;
	Own<HttpRoot::Reader> _data;
	kj::ForkedPromise<std::shared_ptr<kj::HttpServer>> _server;
	kj::ForkedPromise<unsigned int> _port;
	//kj::ForkedPromise<void> _listen;
	kj::ForkedPromise<void> _listen;
};

}