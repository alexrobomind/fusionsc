#include <fsc/local.h>
#include <fsc/services.h>
#include <fsc/data.h>
#include <fsc/networking.h>
#include <fsc/textio-yaml.h>

#include <capnp/rpc-twoparty.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>

#include <iostream>
#include <functional>

#include <capnp/ez-rpc.h>

#include "fsc-tool.h"

using namespace fsc;

namespace {

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

struct ServerTool {
	kj::ProcessContext& context;
	Maybe<uint64_t> port = nullptr;
	kj::String address = kj::heapString("0.0.0.0");
	
	struct ReadFromStdin {};
	OneOf<decltype(nullptr), LocalConfig::Reader, kj::Path, ReadFromStdin> config = nullptr;
	
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
		auto l = newLibrary();
		auto lt = l -> newThread();
		auto& ws = lt->waitScope();
		
		Temporary<LocalConfig> loadedConfig;
		
		if(config.is<kj::Path>()) {
			auto configFile = lt -> filesystem().getCurrent().openFile(config.get<kj::Path>());
			auto configString = configFile -> readAllText();
			textio::load(configFile -> readAllBytes(), *textio::createVisitor(loadedConfig), textio::Dialect::YAML);
		} else if(config.is<LocalConfig::Reader>()) {
			loadedConfig = config.get<LocalConfig::Reader>();
		} else if(config.is<ReadFromStdin>()){
			auto configString = readFromStdin();
			textio::load(configString.asBytes(), *textio::createVisitor(loadedConfig), textio::Dialect::YAML);
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
		listenRequest.setServer(root);
		
		{
			auto info = root.getInfoRequest().send().wait(ws);
			listenRequest.setFallback(kj::heap<NodeInfoProvider>(info));
		}
		
		
		auto openPort = listenRequest.sendForPipeline().getOpenPort();
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
			"FusionSC server\n",
			"Protocol version ", FSC_PROTOCOL_VERSION, "\n"
		);
		
		return kj::MainBuilder(context, infoString, "Creates an FSC server")
			.addOptionWithArg({'a', "address"}, KJ_BIND_METHOD(*this, setAddress), "<address>", "Address to listen on, defaults to 0.0.0.0")
			.addOptionWithArg({'p', "port"}, KJ_BIND_METHOD(*this, setPort), "<port>", "Port to listen on, defaults to system-assigned")
			.addOptionWithArg({'b', "builtin"}, KJ_BIND_METHOD(*this, setBuiltin), "<built-in>", "Name of built-in profile, either 'computeNode' or 'loginNode'")
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
