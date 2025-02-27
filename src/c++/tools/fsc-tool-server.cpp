#include <fsc/local.h>
#include <fsc/services.h>
#include <fsc/data.h>
#include <fsc/networking.h>
#include <fsc/structio-yaml.h>
#include <fsc/data-viewer.h>
#include <fsc/break.h>

#include <capnp/rpc-twoparty.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>
#include <kj/compat/http.h>
#include <kj/compat/url.h>

#include <fsc/dynamic.capnp.h>
#include <fsc/magnetics.capnp.h>
#include <fsc/geometry.capnp.h>
#include <fsc/offline.capnp.h>


#include <iostream>
#include <functional>

#include <capnp/ez-rpc.h>

#include "fsc-tool.h"

using namespace fsc;

namespace {
	
static capnp::SchemaLoader schemaLoader = capnp::SchemaLoader();

struct RootServiceProvider : public NetworkInterface::Listener::Server {
	Temporary<LocalConfig> config;
	
	RootServiceProvider(LocalConfig::Reader configIn) :
		config(configIn)
	{}
	
	Promise<void> accept(AcceptContext ctx) override {
		ctx.initResults().setClient(createRoot(config.asReader()));
		return READY_NOW;
	}
};

struct NodeInfoProvider : public SimpleHttpServer::Server {
	Temporary<NodeInfo> nodeInfo;
	
	NodeInfoProvider(NodeInfo::Reader info) :
		nodeInfo(info)
	{}
	
	Promise<void> serve(ServeContext ctx) override {		
		YAML::Emitter emitter;
		emitter << nodeInfo.asReader();
		
		auto result = kj::str(
			"<html>",
			"	<head><title>FusionSC Server</title></head>",
			"	<body style='font-size: large'>",
			"		<h1>FusionSC Server</h1>",
			"		This is a server for the FusionSC library."
			"		To use it, you need to use the <a href='https://alexrobomind.github.io/fusionsc'>FusionSC library</a>."
			"		<h2>Node information:</h2>",
			"		<code style='white-space: pre-wrap'>", emitter.c_str() ,"</code>"
			"	</body>",
			"</html>"
		);
		
		auto r = ctx.initResults();
		r.setStatus(200);
		r.setBody(result);
		
		return READY_NOW;
	}
};

struct WebFrontend : public kj::HttpService {
	RootService::Client clt;
	
	WebFrontend(RootService::Client clt) : clt(mv(clt)) {}
	
	enum PathType { NO_PATH, DB_NAME, DB_INNER };
	
	kj::Promise<void> request(
		kj::HttpMethod method, kj::StringPtr url, const kj::HttpHeaders& headers,
		kj::AsyncInputStream& requestBody, kj::HttpService::Response& response
	) override {
		auto rootUrl = kj::Url::parse(url, kj::Url::HTTP_REQUEST);
		
		PathType pathType = NO_PATH;
		
		kj::String dbName = nullptr;
		kj::Url dbUrl;
		kj::Vector<kj::String> dbPath;
		for(auto& s : rootUrl.path) {
			if(pathType == NO_PATH && s == "warehouses") {
				pathType = DB_NAME;
				continue;
			}
			
			if(pathType == DB_NAME) {
				dbName = mv(s);
				pathType = DB_INNER;
				continue;
			}
			
			if(pathType == DB_INNER) {
				dbUrl.path.add(mv(s));
			}
		}
		
		if(pathType == NO_PATH) {
			return root(response);
		}
		
		// Open database
		auto dbReq = clt.getWarehouseRequest();
		dbReq.setName(dbName);
		auto wh = dbReq.send().getWarehouse();
		
		// Create data viewer
		auto viewer = createDataViewer(wh, schemaLoader);
		auto result = viewer -> request(method, dbUrl.toString(kj::Url::HTTP_REQUEST), headers, requestBody, response);
		return result.attach(mv(viewer));
	}
	
	Promise<void> root(kj::HttpService::Response& r) {
		return clt.getInfoRequest().send()
		.then([this, &r](auto response) {
			YAML::Emitter emitter;
			emitter << response;
			
			auto result = kj::strTree(
				"<html>",
				"	<head><title>FusionSC Server</title></head>",
				"	<body style='font-size: large'>",
				"		<h1>FusionSC Server</h1>",
				"		This is a server for the FusionSC library."
				"		To use it, you need to use the <a href='https://alexrobomind.github.io/fusionsc'>FusionSC library</a>."
				"		<h2>Node information:</h2>",
				"		<code style='white-space: pre-wrap'>", emitter.c_str() ,"</code>"
			);
			
			if(response.getWarehouses().size() > 0) {
				result = kj::strTree(mv(result), "<h2>Warehouses</h2>This node exposes the following warehouses: <br /><ul>");
				
				for(auto wh : response.getWarehouses())
					result = kj::strTree(mv(result), "<li><a href='warehouses/", wh, "/show'>", wh, "</a></li>");
				
				result = kj::strTree(mv(result), "</ul>");
			}
			
			result = kj::strTree(mv(result), "</body></html>");
			
			return sendText(r, result.flatten());
		});
	}
	
	Promise<void> sendText(kj::HttpService::Response& response, kj::String p) {
		kj::HttpHeaderTable tbl;
		kj::HttpHeaders headers(tbl);
		headers.set(kj::HttpHeaderId::CONTENT_TYPE, "text/html; charset=utf-8");
		
		auto ptr = p.asPtr();
		auto os = response.send(200, "OK", headers);
		auto result = os -> write(ptr.begin(), ptr.size());
	
		return result.attach(mv(os), mv(p));
	}
};

struct ServerTool {
	kj::ProcessContext& context;
	Maybe<uint64_t> port = nullptr;
	kj::String address = kj::heapString("0.0.0.0");
	
	struct ReadFromStdin {};
	OneOf<decltype(nullptr), LocalConfig::Reader, kj::Path, ReadFromStdin> config = nullptr;
	
	Maybe<size_t> ramObjectLimit = nullptr;
	
	ServerTool(kj::ProcessContext& context):
		context(context)
	{}
	
	bool setAddress(kj::StringPtr val) {
		address = kj::str(val);
		return true;
	}
	
	bool setPort(kj::StringPtr val) {
		port = val.parseAs<uint64_t>();
		return true;
	}
	
	bool setBuiltin(kj::StringPtr name) {
		KJ_REQUIRE(config.is<decltype(nullptr)>(), "Can only specify one built-in profile OR settings file");
		
		if(name == "loginNode")
			config = LOGIN_NODE_PROFILE.get();
		else if(name == "computeNode")
			config = COMPUTE_NODE_PROFILE.get();
		else {
			KJ_FAIL_REQUIRE("Invalid profile name, must be 'loginNode' or 'computeNode'", name);
		}
		
		return true;
	}
	
	bool setFile(kj::StringPtr fileName) {
		KJ_REQUIRE(config.is<decltype(nullptr)>(), "Can only specify one built-in profile OR settings file");
		config = kj::Path(nullptr).evalNative(fileName);
		return true;
	}
	
	bool setReadStdin() {
		config = ReadFromStdin();
		return true;
	}
	
	bool setRamObjectLimit(kj::StringPtr val) {
		ramObjectLimit = val.parseAs<size_t>();
		return true;
	}
	
	kj::String readFromStdin() {
		std::cout << "Reading YAML configuration from stdin (console). Please end your configuration with either ... or --- (YAML document termination markers)" << std::endl << std::endl;
		
		kj::StringTree yamlDoc;
		
		while(true) {
			std::string stdLine;
			std::getline(std::cin, stdLine);
			
			kj::StringPtr line(stdLine.c_str());
			
			if(line.startsWith("...") || line.startsWith("---")) {
				break;
			}
			
			yamlDoc = kj::strTree(mv(yamlDoc), "\n", line);
			
			if(std::cin.eof())
				break;
		}
		
		return yamlDoc.flatten();
	}
		
	bool run() {
		BreakHandler breakHandler;
		
		auto l = newLibrary();
		auto lt = l -> newThread();
		auto& ws = lt->waitScope();
		
		KJ_IF_MAYBE(pLimit, ramObjectLimit) {
			LocalDataService::Limits limitsStruct;
			limitsStruct.maxRAMObjectSize = *pLimit;
			lt -> dataService().setLimits(limitsStruct);
		}

		// Prepare schema loader
		schemaLoader.loadCompiledTypeAndDependencies<MagneticField>();
		schemaLoader.loadCompiledTypeAndDependencies<Geometry>();
		schemaLoader.loadCompiledTypeAndDependencies<DynamicObject>();
		schemaLoader.loadCompiledTypeAndDependencies<OfflineData>();
		schemaLoader.loadCompiledTypeAndDependencies<Mesh>();
		schemaLoader.loadCompiledTypeAndDependencies<MergedGeometry>();
		
		Temporary<LocalConfig> loadedConfig;
		
		if(config.is<kj::Path>()) {
			auto configFile = lt -> filesystem().getCurrent().openFile(config.get<kj::Path>());
			auto configString = configFile -> readAllText();
			structio::load(configFile -> readAllBytes(), *structio::createVisitor(loadedConfig), structio::Dialect::YAML);
		} else if(config.is<LocalConfig::Reader>()) {
			loadedConfig = config.get<LocalConfig::Reader>();
		} else if(config.is<ReadFromStdin>()){
			auto configString = readFromStdin();
			structio::load(configString.asBytes(), *structio::createVisitor(loadedConfig), structio::Dialect::YAML);
		}
		
		// Dump configuration to console
		std::cout << " --- Configuration --- " << std::endl << std::endl;
		YAML::Emitter emitter(std::cout);
		emitter << loadedConfig.asReader();
		std::cout << std::endl << std::endl;
		
		NetworkInterface::Client networkInterface = kj::heap<LocalNetworkInterface>();
		
		auto listenRequest = networkInterface.serveRequest();
		KJ_IF_MAYBE(pPort, port) {
			listenRequest.setPortHint(*pPort);
		}
		listenRequest.setHost(address);
		
		RootService::Client root = createRoot(loadedConfig.asReader());
		/* listenRequest.setServer(root);
		
		{
			auto info = root.getInfoRequest().send().wait(ws);
			listenRequest.setFallback(kj::heap<NodeInfoProvider>(info));
		}*/

		auto& network = lt -> network();
		unsigned int portHint = 0;
		KJ_IF_MAYBE(pPort, port) {
			portHint = *pPort;
		}
		
		/*auto openPort = listenRequest.sendForPipeline().getOpenPort();
		 */

		auto parsedAddress = network.parseAddress(address, portHint).wait(ws);
		auto openPort = listenViaHttp(parsedAddress -> listen(), root, kj::heap<WebFrontend>(root)); // createDataViewer(root, schemaLoader));
		auto info = openPort.getInfoRequest().send().wait(ws);
		
		std::cout << "Serving protocol version " << FSC_PROTOCOL_VERSION << std::endl;
		std::cout << "Listening on port " << info.getPort() << std::endl;
		
		breakHandler.onBreak().wait(ws);
		
		std::cout << "Draining ..." << std::endl;
		
		openPort.stopListeningRequest().send().wait(ws);
		
		try {
			breakHandler.wrap(openPort.drainRequest().send()).wait(ws);
		} catch(kj::Exception& ) {
		}
		
		std::cout << "Shutting down" << std::endl;		
		return true;
	}
	
	auto getMain() {
		auto infoString = kj::str(
			"FusionSC server\n",
			"Protocol version ", FSC_PROTOCOL_VERSION, "\n"
		);
		
		return kj::MainBuilder(context, infoString, "Creates an FSC server")
			.addOptionWithArg({'a', "address"}, KJ_BIND_METHOD(*this, setAddress), "<address>", "Address to listen on, defaults to 0.0.0.0")
			.addOptionWithArg({'p', "port"}, KJ_BIND_METHOD(*this, setPort), "<port>", "Port to listen on, defaults to system-assigned")
			.addOptionWithArg({'b', "builtin"}, KJ_BIND_METHOD(*this, setBuiltin), "<built-in>", "Name of built-in profile, either 'computeNode' or 'loginNode'")
			.addOptionWithArg({"ramObjectLimit"}, KJ_BIND_METHOD(*this, setRamObjectLimit), "<RAM object size limit>", "Object size limit above which objects are allocated in memory-mapped files")
			.addOption({"stdin"}, KJ_BIND_METHOD(*this, setReadStdin), "Read configuration from stdin")
			.expectOptionalArg("<settings file>", KJ_BIND_METHOD(*this, setFile))
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build();
	}
};

}

fsc_tool::MainGen fsc_tool::server(kj::ProcessContext& ctx) {
	return KJ_BIND_METHOD(ServerTool(ctx), getMain);
}
