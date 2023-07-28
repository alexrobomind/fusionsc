#include <fsc/local.h>
#include <fsc/services.h>
#include <fsc/data.h>
#include <fsc/networking.h>
#include <fsc/yaml.h>
#include <fsc/load-balancer.h>

#include <capnp/rpc-twoparty.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>

#include <iostream>
#include <functional>

#include <capnp/ez-rpc.h>

using namespace fsc;

struct MainCls {
	kj::ProcessContext& context;
	Maybe<uint64_t> port = nullptr;
	kj::String address = kj::heapString("0.0.0.0");
	
	OneOf<decltype(nullptr), kj::Path> config = nullptr;
	
	MainCls(kj::ProcessContext& context):
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
	
	bool setFile(kj::StringPtr fileName) {
		KJ_REQUIRE(config.is<decltype(nullptr)>(), "Can only specify one built-in profile OR settings file");
		config = kj::Path(nullptr).evalNative(fileName);
		return true;
	}
		
	bool run() {
		auto l = newLibrary(true);
		auto lt = l -> newThread();
		auto& ws = lt->waitScope();
		
		Temporary<LoadBalancerConfig> loadedConfig;
		
		{
			YAML::Node root;
			
			if(config.is<kj::Path>()) {
				auto configFile = lt -> filesystem().getCurrent().openFile(config.get<kj::Path>());
				auto configString = configFile -> readAllText();
				root = YAML::Load(configString.cStr());
			} else {
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
				
				root = YAML::Load(yamlDoc.flatten().cStr());
			}
			
			load(loadedConfig, root);
		}
		
		// Dump configuration to console
		std::cout << " --- Configuration --- " << std::endl << std::endl;
		YAML::Emitter emitter(std::cout);
		emitter << loadedConfig.asReader();
		std::cout << std::endl << std::endl;
				
		NetworkInterface::Client networkInterface = kj::heap<LocalNetworkInterface>();
		
		// Create load balancer
		auto loadBalancer = newLoadBalancer(networkInterface, loadedConfig);
		
		auto serveRequest = networkInterface.serveRequest();
		KJ_IF_MAYBE(pPort, port) {
			serveRequest.setPortHint(*pPort);
		}
		serveRequest.setHost(address);
		serveRequest.setServer(mv(loadBalancer));
		
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
			"FusionSC load balancer\n",
			"Protocol version ", FSC_PROTOCOL_VERSION, "\n"
		);
		
		return kj::MainBuilder(context, infoString, "Creates a load balancer for fusionsc services")
			.addOptionWithArg({'a', "address"}, KJ_BIND_METHOD(*this, setAddress), "<address>", "Address to listen on, defaults to 0.0.0.0")
			.addOptionWithArg({'p', "port"}, KJ_BIND_METHOD(*this, setPort), "<port>", "Port to listen on, defaults to system-assigned")
			.expectOptionalArg("<settings file>", KJ_BIND_METHOD(*this, setFile))
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build();
	}
};

KJ_MAIN(MainCls)
