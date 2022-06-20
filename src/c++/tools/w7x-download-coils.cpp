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
	static constexpr auto DEFAULT_COILSDB_ADDRESS = "http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest";
	static constexpr auto DEFAULT_COMPSDB_ADDRESS = "http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDBRest";
	
	// Fields initialized in the constructor
	kj::ProcessContext& context;
	Library l;
	LibraryThread lt;
	kj::WaitScope& ws;
	
	// Fields with default values
	template<typename T>
	using ResultPromise = Promise<capnp::Response<T>>;
	
	kj::TreeMap<unsigned int, ResultPromise<Filament>> coils;
	kj::TreeMap<unsigned int, ResultPromise<CoilsDBConfig>> configs;
	
	kj::TreeMap<unsigned int, ResultPromise<Mesh>> meshes;
	kj::TreeMap<unsigned int, ResultPromise<ComponentsDB::GetAssemblyResults>> assemblies;
	
	CoilsDB::Client coilsDB = nullptr;
	ComponentsDB::Client componentsDB = nullptr;
	
	capnp::MallocMessageBuilder resultBuilder;
	
	kj::String coilsAddress = kj::heapString(DEFAULT_COILSDB_ADDRESS);
	kj::String compsAddress = kj::heapString(DEFAULT_COMPSDB_ADDRESS);
	
	Own<PromiseFulfiller<CoilsDB::Client>>      coilsFulfiller;
	Own<PromiseFulfiller<ComponentsDB::Client>> compsFulfiller;
	
	kj::String outputName = kj::heapString("w7x-offline.fsc");
	
	MainCls(kj::ProcessContext& context):
		context(context),
		l(newLibrary()),
		lt(l -> newThread()),
		ws(lt -> waitScope())
	{		
		auto coilsPaf = kj::newPromiseAndFulfiller<CoilsDB::Client>();
		auto compsPaf = kj::newPromiseAndFulfiller<ComponentsDB::Client>();
		
		coilsDB = CoilsDB::Client(mv(coilsPaf.promise));
		coilsFulfiller = mv(coilsPaf.fulfiller);
		
		componentsDB = ComponentsDB::Client(mv(compsPaf.promise));
		compsFulfiller = mv(compsPaf.fulfiller);
	}
	
	bool setCoilsDBAddress(kj::StringPtr str) {
		coilsAddress = kj::heapString(str);
		return true;
	}
	
	bool setCompsDBAddress(kj::StringPtr str) {
		compsAddress = kj::heapString(str);
		return true;
	}
	
	bool addCoil(kj::StringPtr str) {
		downloadCoil(str.parseAs<unsigned int>());
		return true;
	}
	
	bool addConfig(kj::StringPtr str) {
		downloadConfig(str.parseAs<unsigned int>());
		return true;
	}
	
	bool addMesh(kj::StringPtr str) {
		downloadMesh(str.parseAs<unsigned int>());
		return true;
	}
	
	bool addAssembly(kj::StringPtr str) {
		downloadAssembly(str.parseAs<unsigned int>());
		return true;
	}
	
	bool setOutput(kj::StringPtr str) {
		outputName = kj::heapString(str);
		return true;
	}
	
	bool addDefault() {
		// Add default stuff here
		for(auto i : kj::Range<unsigned int>(160, 230))
			downloadCoil(i);
		return true;
	}
	
	void downloadCoil(unsigned int id) {
		KJ_IF_MAYBE(cPtr, coils.find(id)) {
			printf("Coil %d already requested\n", id);
			return;
		}
		
		auto request = coilsDB.getCoilRequest();
		request.setId(id);
		
		printf("Downloading coil %d\n", id);
		coils.insert(id, request.send().eagerlyEvaluate(nullptr));
	}
	
	void downloadMesh(unsigned int id) {
		KJ_IF_MAYBE(mPtr, meshes.find(id)) {
			printf("Mesh %d already requested\n", id);
			return;
		}
		
		auto request = componentsDB.getMeshRequest();
		request.setId(id);
		
		printf("Downloading mesh %d\n", id);
		meshes.insert(id, request.send().eagerlyEvaluate(nullptr));
	}
	
	void downloadAssembly(unsigned int id) {
		KJ_IF_MAYBE(cPtr, assemblies.find(id)) {
			printf("Assembly %d already requested\n", id);
			return;
		}
		
		auto request = componentsDB.getAssemblyRequest();
		request.setId(id);
		
		auto result = request.send()
		.then([id, this](auto response) {
			for(auto component : response.getComponents())
				downloadMesh(component);
			
			return mv(response);
		}).eagerlyEvaluate(nullptr);
		
		printf("Downloading assembly %d\n", id);
		assemblies.insert(id, mv(result));
	}
	
	void downloadConfig(unsigned int id) {
		KJ_IF_MAYBE(cPtr, configs.find(id)) {
			printf("Config %d already requested\n", id);
			return;
		}
		
		auto request = coilsDB.getConfigRequest();
		request.setId(id);
		
		auto result = request.send()
		.then([id, this](auto response) {
			for(auto coil : response.getCoils())
				downloadCoil(coil);
			
			return mv(response);
		}).eagerlyEvaluate(nullptr);
		
		printf("Downloading config %d\n", id);
		configs.insert(id, mv(result));
	}
	
	bool run() {
		// Set up client
		coilsFulfiller->fulfill(newCoilsDBFromWebservice(mv(coilsAddress), lt));
		compsFulfiller->fulfill(newComponentsDBFromWebservice(mv(compsAddress), lt));
		
		Temporary<OfflineData> output;
		
		// Download configurations
		auto nConfigs = configs.size();
		auto outConfigs = output.initW7xConfigs(nConfigs);
		
		unsigned int i = 0;
		for(auto it = configs.begin(); i < nConfigs; ++i, ++it) {
			auto outConfig = outConfigs[i];
			
			outConfig.setId(it->key);
			outConfig.setConfig(it->value.wait(ws));
		}
		
		// Download assemblies
		auto nAssemblies = assemblies.size();
		auto outAssemblies = output.initW7xAssemblies(nAssemblies);
		
		i = 0;
		for(auto it = assemblies.begin(); i < nAssemblies; ++i, ++it) {
			auto outAssembly = outAssemblies[i];
			
			auto result = it->value.wait(ws);
			
			outAssembly.setId(it->key);
			outAssembly.setAssembly(result.getComponents());
		}
		
		// Download coils
		auto nCoils = coils.size();
		auto outCoils = output.initW7xCoils(nCoils);
		
		i = 0;
		for(auto it = coils.begin(); i < nCoils; ++i, ++it) {
			auto outCoil = outCoils[i];
			
			outCoil.setId(it->key);
			outCoil.setFilament(it->value.wait(ws));
		}
		
		// Download meshes
		auto nMeshes = meshes.size();
		auto outMeshes = output.initW7xComponents(nMeshes);
		
		i = 0;
		for(auto it = meshes.begin(); i < nMeshes; ++i, ++it) {
			auto outMesh = outMeshes[i];
			
			outMesh.setId(it->key);
			outMesh.setComponent(it->value.wait(ws));
		}
		
		// Write archive
		auto filePath = kj::Path::parse(outputName);
		auto file = lt->filesystem().getCurrent().openFile(filePath, kj::WriteMode::CREATE | kj::WriteMode::MODIFY | kj::WriteMode::CREATE_PARENT);
		
		LocalDataService& ds = lt->dataService();
		ds.writeArchive(
			ds.publish(lt->randomID(), output.asReader()),
			*file
		).wait(ws);
		
		return true;
	}
	
	auto getMain() {
		return kj::MainBuilder(context, "Coils downloader", "Downloads W7-X coils into OfflineData database")
			.addOptionWithArg({"coilsdb"}, KJ_BIND_METHOD(*this, setCoilsDBAddress), "<address>", "Address of CoilsDB")
			.addOptionWithArg({"compsdb"}, KJ_BIND_METHOD(*this, setCompsDBAddress), "<address>", "Address of ComponentsDB")
			.addOptionWithArg({"coil"}, KJ_BIND_METHOD(*this, addCoil), "<coilID>", "Download a single coil")
			.addOptionWithArg({"config"}, KJ_BIND_METHOD(*this, addConfig), "<coilID>", "Download a single coil")
			.addOptionWithArg({"mesh"}, KJ_BIND_METHOD(*this, addMesh), "<coilID>", "Download a single coil")
			.addOptionWithArg({"assembly"}, KJ_BIND_METHOD(*this, addAssembly), "<coilID>", "Download a single coil")
			.addOption({"default"}, KJ_BIND_METHOD(*this, addDefault), "Add default payloads (CAD coils, baseline configurations, most-used PFCs)")
			.addOptionWithArg({"-o", "output"}, KJ_BIND_METHOD(*this, setOutput), "<output file name>", "Specify output file")
			.callAfterParsing(KJ_BIND_METHOD(*this, run))
			.build();
	}
};

KJ_MAIN(MainCls)
