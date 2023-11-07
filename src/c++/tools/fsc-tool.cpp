#include "fsc-tool.h"

#include <fsc/services.h>

using namespace fsc;

struct MainCls {
	kj::ProcessContext& context;
	
	MainCls(kj::ProcessContext& context):
		context(context)
	{}
	
	kj::MainFunc getMain() {
		auto infoString = kj::str(
			"FusionSC tool\n",
			"Protocol version ", FSC_PROTOCOL_VERSION, "\n"
		);
		
		return kj::MainBuilder(context, infoString, "Multi-purpose command-line tool for fusionsc related functionality")
			.addSubCommand("server", fsc_tool::server(context), "Remotely accessible fusionsc node")
			.addSubCommand("worker", fsc_tool::worker(context), "Worker node that registers itself at a remote server")
			.addSubCommand("warehouse", fsc_tool::warehouse(context), "Serves and maintains object warehouses (databases)")
			.addSubCommand("load-balancer", fsc_tool::loadBalancer(context), "Provides a load-balancing tool")
			.addSubCommand("capnp", fsc_tool::capnp(context), "Tool for Cap'n'proto related operations")
			.build()
		;
	}
};

KJ_MAIN(MainCls)