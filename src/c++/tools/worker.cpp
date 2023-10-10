#include <fsc/local.h>
#include <fsc/services.h>
#include <fsc/data.h>
#include <fsc/networking.h>
#include <fsc/yaml.h>
#include <fsc/matcher.h>

#include <capnp/rpc-twoparty.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>
#include <kj/encoding.h>

#include <iostream>
#include <functional>

#include <capnp/ez-rpc.h>

using namespace fsc;

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

struct MainCls {
	kj::ProcessContext& context;
	
	kj::String url = kj::heapString("http://localhost");
kj::Array<const byte> token = kj::heapArray<const byte>({0});
	
	struct ReadFromStdin {};
	OneOf<decltype(nullptr), LocalConfig::Reader, kj::Path, ReadFromStdin> config = nullptr;
	
	MainCls(kj::ProcessContext& context):
		context(context)
	{}
	
	bool setUrl(kj::StringPtr val) {
		url = kj::str(val);
		return true;
	}
	
	bool setToken(kj::StringPtr val) {
		token = kj::decodeBase64(val);
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
			YAML::Node root = YAML::Load(configString.cStr());
			load(loadedConfig, root);
		} else if(config.is<LocalConfig::Reader>()) {
			loadedConfig = config.get<LocalConfig::Reader>();
		} else if(config.is<ReadFromStdin>()){
			auto configString = readFromStdin();
			YAML::Node root = YAML::Load(configString.cStr());
			load(loadedConfig, root);
		}
		
		// Dump configuration to console
		std::cout << " --- Configuration --- " << std::endl << std::endl;
		YAML::Emitter emitter(std::cout);
		emitter << loadedConfig.asReader();
		std::cout << std::endl << std::endl;
		
		// Create backend
		auto localRoot = createRoot(loadedConfig.asReader());
		
		// Create listener for destruction
		auto disconnectEvent = kj::newPromiseAndFulfiller<void>();
		localRoot = localRoot.attach(kj::defer([f = mv(disconnectEvent.fulfiller)]() mutable {
			f -> fulfill();
		}));
		
		// Connect to remote interface
		std::cout << "Connecting ..." << std::endl;
		NetworkInterface::Client networkInterface = kj::heap<LocalNetworkInterface>();
		auto connectRequest = networkInterface.connectRequest();
		connectRequest.setUrl(url);
		
		auto connection = connectRequest.send().wait(ws).getConnection();
		
		std::cout << "Publishing interface ..." << std::endl;
		auto matcher = connection.getRemoteRequest().sendForPipeline().getRemote().castAs<RootService>().matcherRequest().sendForPipeline().getService();
		auto putRequest = matcher.putRequest();
		putRequest.setToken(token.asPtr());
		putRequest.setCap(mv(localRoot));
		putRequest.send().wait(ws);
		
		std::cout << "Waiting for completion ..." << std::endl;
		disconnectEvent.promise.wait(ws);
		
		std::cout << "Shutting down" << std::endl;		
		return true;
	}
	
	auto getMain() {
		auto infoString = kj::str(
			"FusionSC worker node\n",
			"Protocol version ", FSC_PROTOCOL_VERSION, "\n"
		);
		
		return kj::MainBuilder(context, infoString, "Creates a fusionsc worker")
			.expectArg("<Upstream URL>", KJ_BIND_METHOD(*this, setUrl))
			.expectArg("<Token>", KJ_BIND_METHOD(*this, setToken))
			.expectOptionalArg("<settings file>", KJ_BIND_METHOD(*this, setFile))
			.addOptionWithArg({'b', "builtin"}, KJ_BIND_METHOD(*this, setBuiltin), "<built-in>", "Name of built-in profile, either 'computeNode' or 'loginNode'")
			.addOption({"stdin"}, KJ_BIND_METHOD(*this, setReadStdin), "Read configuration from stdin")
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build();
	}
};

KJ_MAIN(MainCls)
