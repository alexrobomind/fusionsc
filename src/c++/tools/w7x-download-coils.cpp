#include <fsc/offline.capnp.h>

#include <fsc/common.h>
#include <fsc/local.h>

#include <kj/main.h>
#include <fsc/devices/w7x.h>

#include <iostream>

#include <list>
#include <kj/vector.h>

using namespace fsc;
using namespace fsc::devices::w7x;

struct MainCls {
	static constexpr auto DEFAULT_ADDRESS = "http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest/coil/";
	
	// Fields initialized in the constructor
	kj::ProcessContext& context;
	Library l;
	LibraryThread lt;
	kj::WaitScope& ws;
	
	// Fields with default values	
	kj::Vector<unsigned int> coils;
	CoilsDB::Client coilsDB = nullptr;
	capnp::MallocMessageBuilder resultBuilder;
	kj::String address = kj::heapString(DEFAULT_ADDRESS);
	
	MainCls(kj::ProcessContext& context):
		context(context),
		l(newLibrary()),
		lt(l -> newThread()),
		ws(lt -> waitScope())
	{
		setAddress(DEFAULT_ADDRESS);
	}
	
	bool setAddress(kj::StringPtr str) {
		address = lt -> network().parseAddress(str).wait(ws);
		return true;
	}
	
	bool addCoil(kj::StringPtr str) {
		coils.add(str.parseAs<unsigned int>());
		return true;
	}
	
	bool addRange(kj::StringPtr str) {
		auto n = str.parseAs<unsigned int>();
		KJ_REQUIRE(coils.size() > 0, "Need existing coil id to build range from");
		auto start = coils[coils.size() - 1];
		
		for(auto i = start + 1; i < start + n; ++i)
			coils.add(i);
		
		return true;
	}
	
	bool run() {
		// Connect
		auto coilsDB = newCoilsDBFromWebservice(mv(address), lt);
		
		auto root = resultBuilder.initRoot<OfflineData>();
		unsigned int n_coils = coils.size();
		auto output = root.initW7xCoils(n_coils);
		auto downloadTasks = kj::heapArrayBuilder<Promise<void>>(n_coils);
		
		for(size_t i = 0; i < coils.size(); ++i) {
			downloadTasks.add(downloadCoil(coils[i], output[i]));
		}
		
		// Wait for download to finish
		kj::joinPromises(downloadTasks.finish()).wait(ws);
		return true;
	}
	
	Promise<void> downloadCoil(unsigned int i, OfflineData::W7XCoil::Builder output) {
		auto request = coilsDB.getCoilRequest();
		request.setId(i);
		
		auto response = request.send();
		return response.then([output, i](auto response) mutable {
			std::cout << "Received coil " << i << std::endl;
			output.setId(i);
			output.setFilament(response.getFilament());
		});
	}
	
	auto getMain() {
		return kj::MainBuilder(context, "Coils downloader", "Downloads W7-X coils into OfflineData database")
			.addOptionWithArg({"--coilsdb"}, KJ_BIND_METHOD(*this, setAddress), "<address>", "Address of CoilsDB")
			.addOptionWithArg({"--coilsdb"}, KJ_BIND_METHOD(*this, setAddress), "<address>", "Address of CoilsDB")
			.addOptionWithArg({"-n"}, KJ_BIND_METHOD(*this, addRange), "<range>", "Extend previous range of coils to encompass additional <range>-1 coils.")
			.expectOneOrMoreArgs("<coil>", KJ_BIND_METHOD(*this, addCoil))
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build();
	}
};

KJ_MAIN(MainCls)