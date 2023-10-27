#include "fsc-tool.h"

#include <fsc/services.h>

static struct MainCls {
	kj::ProcessContext& context;
	
	MainCls(kj::ProcessContext& context):
		context(context)
	{}
	
	kj::MainFunc getMain() {
		auto infoString = kj::str(
			"FusionSC tool\n",
			"Protocol version ", FSC_PROTOCOL_VERSION, "\n"
		);
		
		return kj::MainBuilder(context, infoString, "Multi-purpose tool for fusionsc related functionality")
			
	}
}