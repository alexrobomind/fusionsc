#include <fsc/local.h>
#include <fsc/networking.h>
#include <fsc/services.h>
#include <fsc/sqlite.h>
#include <fsc/odb.h>

#include <capnp/rpc-twoparty.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>
#include <kj/compat/url.h>

#include <iostream>
#include <functional>

#include <capnp/ez-rpc.h>

using namespace fsc;

struct SimpleMessageFallback : public SimpleHttpServer::Server {
	Promise<void> serve(ServeContext ctx) {
		unsigned int UPGRADE_REQUIRED = 426;
		
		auto results = ctx.initResults();
		results.setStatus(UPGRADE_REQUIRED);
		results.setStatusText("Upgrade required");
		results.setBody(
			"This is a FusionSC warehouse, which expects WebSocket requests."
			" Please connect to it using the FusionSC client."
			" See: https://jugit.fz-juelich.de/a.knieps/fsc for more information"
		);
		return READY_NOW;
	}
};

struct MainCls {
	kj::ProcessContext& context;
	Maybe<uint64_t> port = nullptr;
	kj::String address = kj::heapString("0.0.0.0");
	
	kj::String dbFile;
	kj::String tablePrefix;
	
	kj::String rootPath;
	
	bool writeAccess = false;
	
	MainCls(kj::ProcessContext& context):
		context(context), tablePrefix(kj::heapString("warehouse")), rootPath(kj::heapString(""))
	{}
	
	bool setAddress(kj::StringPtr val) {
		address = kj::str(val);
		return true;
	}
	
	bool setPort(kj::StringPtr val) {
		port = val.parseAs<uint64_t>();
		return true;
	}
	
	bool setWriteAccess() {
		writeAccess = true;
		return true;
	}
	
	bool setDb(kj::StringPtr file) {
		dbFile = kj::heapString(file);
		return true;
	}
	
	bool setTablePrefix(kj::StringPtr prefix) {
		tablePrefix = kj::heapString(prefix);
		return true;
	}
	
	bool setPath(kj::StringPtr pathStr) {
		rootPath = kj::heapString(pathStr);
		return true;
	}
		
	bool run() {
		auto l = newLibrary();
		auto lt = l -> newThread();
		auto& ws = lt->waitScope();
				
		// Open database
		
		bool readOnly = !writeAccess;
		auto conn = connectSqlite(dbFile, readOnly);
		auto db = ::fsc::openWarehouse(*conn, tablePrefix, readOnly);
		auto root = db.getRootRequest().sendForPipeline().getRoot();
		root.whenResolved().wait(ws);
		
		// Get root path
		if(readOnly) {
			auto getRequest = root.getRequest();
			getRequest.setPath(rootPath);
			
			auto rootObject = getRequest.send().wait(ws);
			KJ_REQUIRE(rootObject.isFolder(), "Requested root object is not a folder");
			
			root = rootObject.getFolder();
		} else {
			auto mkdirRequest = root.mkdirRequest();
			mkdirRequest.setPath(rootPath);
			root = mkdirRequest.send().wait(ws).getFolder();
		}
		
		// Create network interface
		NetworkInterface::Client networkInterface = kj::heap<LocalNetworkInterface>();
		
		auto serveRequest = networkInterface.serveRequest();
		KJ_IF_MAYBE(pPort, port) {
			serveRequest.setPortHint(*pPort);
		}
		serveRequest.setHost(address);
		serveRequest.setServer(root);
		serveRequest.setFallback(kj::heap<SimpleMessageFallback>());
		
		auto openPort = serveRequest.sendForPipeline().getOpenPort();
		auto info = openPort.getInfoRequest().send().wait(ws);
		
		std::cout << "Serving protocol version " << FSC_PROTOCOL_VERSION << std::endl;
		std::cout << "Listening on port " << info.getPort() << std::endl;
		
		while(true) {
			try {
				lt -> timer().afterDelay(1 * kj::SECONDS).wait(ws);
			} catch(kj::Exception e) {
				KJ_DBG("Received exception", e);
				break;
			}
		}
		
		std::cout << "Draining ..." << std::endl;
		openPort.drainRequest().send().wait(ws);
		
		std::cout << "Shutting down" << std::endl;		
		return true;
	}
	
	auto getMain() {
		auto infoString = kj::str(
			"FusionSC database server\n",
			"Protocol version ", FSC_PROTOCOL_VERSION, "\n"
		);
		
		return kj::MainBuilder(context, infoString, "Creates an FSC server")
			.addOptionWithArg({'a', "address"}, KJ_BIND_METHOD(*this, setAddress), "<address>", "Address to listen on, defaults to 0.0.0.0")
			.addOptionWithArg({'p', "port"}, KJ_BIND_METHOD(*this, setPort), "<port>", "Port to listen on, defaults to system-assigned")
			
			.addOption({'w', "write-access"}, KJ_BIND_METHOD(*this, setWriteAccess), "Enables write access to the target database")
			.addOptionWithArg({"table-prefix"}, KJ_BIND_METHOD(*this, setTablePrefix), "<prefix>", "Prefix to use for table names (default 'warehouse')")
			.addOptionWithArg({"path"}, KJ_BIND_METHOD(*this, setPath), "<path>", "Path to share relative to database root")
			.expectArg("<database file>", KJ_BIND_METHOD(*this, setDb))
			
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build();
	}
};

KJ_MAIN(MainCls)
