#include <kj/compat/http.h>
#include <capnp/compat/json.h>

#include "w7x.h"

namespace fsc {

namespace devices { namespace w7x {

// === class CoilsDBResolver ===
	
CoilsDBResolver::CoilsDBResolver(LibraryThread& lt, CoilsDB::Client backend) :
	FieldResolverBase(lt),
	backend(backend)
{}

Promise<void> CoilsDBResolver::processField(MagneticField::Reader input, MagneticField::Builder output, ResolveContext context) {
	switch(input.which()) {
		case MagneticField::W7X_MAGNETIC_CONFIG: {
			auto w7xConfig = input.getW7xMagneticConfig();
			
			switch(w7xConfig.which()) {
				case MagneticField::W7xMagneticConfig::COILS_AND_CURRENTS:
					return coilsAndCurrents(w7xConfig.getCoilsAndCurrents(), output, context);
				case MagneticField::W7xMagneticConfig::CONFIGURATION_D_B: {
					auto cdbConfig = w7xConfig.getConfigurationDB();
					//TODO: This should probably be cached
					auto request = backend.getConfigRequest();
					request.setId(cdbConfig.getConfigID());
					auto coilsDBResponse = request.send();
					
					return coilsDBResponse.then([this, output, cdbConfig](auto response) mutable {
						auto config = response.getConfig();
						
						auto coils = config.getCoils();
						auto currents = config.getCurrents();
						
						double scale = config.getScale();
						
						unsigned int n_coils = coils.size();
						KJ_REQUIRE(currents.size() == n_coils);
						
						auto sum = output.initSum(n_coils);
						for(unsigned int i = 0; i < n_coils; ++i) {
							auto filField = sum[i].initFilamentField();
							filField.setCurrent(scale * currents[i]);
							filField.setWidth(cdbConfig.getCoilWidth());
							filField.initFilament().setRef(getCoil(coils[i]));
						}
					});
				}
				default:
					throw "Unknown W7-X configuration";
			}
		}
		default:
			return FieldResolverBase::processField(input, output, context);
	}
}

Promise<void> CoilsDBResolver::coilsAndCurrents(MagneticField::W7xMagneticConfig::CoilsAndCurrents::Reader reader, MagneticField::Builder output, ResolveContext context) {
	output.initSum(N_MAIN_COILS + N_TRIM_COILS + N_CONTROL_COILS);
	
	// Note: The following two statements must not be inlined into one
	// The LocalDataRef must be kept alive in the method
	// so that the underlying data are not deleted.
	LocalDataRef<CoilFields> coilFields = getCoilFields(reader.getCoils());
	CoilFields::Reader fieldsReader = coilFields.get();
	
	using FieldRef = DataRef<MagneticField>::Client;
	unsigned int offset = 0;
	
	auto addOutput = [&](FieldRef field, double current) {
		auto subField = output.getSum()[offset++];
		auto scaleBy = subField.initScaleBy();
		scaleBy.setFactor(current);
		scaleBy.initField().setRef(field);				
	};
	
	for(unsigned int i = 0; i < 5; ++i) {
		addOutput(fieldsReader.getMainCoils()[i], reader.getNonplanar()[i]);
	}
	
	for(unsigned int i = 0; i < 2; ++i) {
		addOutput(fieldsReader.getMainCoils()[i + 5], reader.getPlanar()[i]);
	}
	
	KJ_ASSERT(offset == N_MAIN_COILS);
	
	for(unsigned int i = 0; i < N_TRIM_COILS; ++i) {
		addOutput(fieldsReader.getTrimCoils()[i], reader.getTrim()[i]);
	}
	
	//TODO: Extend to the 2 and 5 cases
	KJ_REQUIRE(reader.getControl().size() == 10);
	
	for(unsigned int i = 0; i < N_CONTROL_COILS; ++i) {
		addOutput(fieldsReader.getControlCoils()[i], reader.getControl()[i]);
	}
	
	KJ_ASSERT(offset == output.getSum().size());
	return kj::READY_NOW;
}

LocalDataRef<CoilFields> CoilsDBResolver::getCoilFields(W7XCoilSet::Reader reader) {
	ID id = ID::fromReader(reader);
	
	KJ_IF_MAYBE(coilPack, coilPacks.find(id)) {
		return *coilPack;
	}
	
	capnp::MallocMessageBuilder tmp;
	auto coilPack = tmp.initRoot<CoilFields>();
	buildCoilFields(reader, coilPack);
	
	auto ref = lt->dataService().publish(id, coilPack);
	coilPacks.insert(id, ref);
	return ref;
}

Promise<void> CoilsDBResolver::processFilament(Filament     ::Reader input, Filament     ::Builder output, ResolveContext context) {
	switch(input.which()) {
		case Filament::W7X_COILS_D_B:
			output.setRef(getCoil(input.getW7xCoilsDB()));
			return kj::READY_NOW;
			
		default:
			return FieldResolverBase::processFilament(input, output, context);
	}
}

DataRef<Filament>::Client CoilsDBResolver::getCoil(uint64_t cdbID) {
	KJ_IF_MAYBE(coilsRef, coils.findEntry(cdbID)) {
		return coilsRef->value;
	}
	 
	auto coilRequest = backend.getCoilRequest();
	coilRequest.setId(cdbID);
	
	DataRef<Filament>::Client newCoil = coilRequest.send().then([cdbID, this](auto response) {
		auto filament = response.getFilament();
		auto ref = lt -> dataService().publish(lt -> randomID(), filament);
		
		return ref;
	});
	coils.insert(cdbID, newCoil);
	// TODO: Install an error handler that clears the entry if the request fails
	return newCoil;
}

void CoilsDBResolver::buildCoilFields(W7XCoilSet::Reader reader, CoilFields::Builder coilPack) {	
	auto getMainCoil = [=](unsigned int i_mod, unsigned int i_coil) -> DataRef<Filament>::Client {
		if(reader.isCoilsDBSet()) {
			unsigned int offset = reader.getCoilsDBSet().getMainCoilOffset();
			unsigned int coilID = i_coil < 5
				? offset + 5 * i_mod + i_coil
				: offset + 2 * i_mod + i_coil + 50;
			return getCoil(coilID);
		} else {
			KJ_REQUIRE(reader.isCustomCoilSet());
			return reader.getCustomCoilSet().getMainCoils()[N_MAIN_COILS * i_mod + i_coil];
		}
	};
	
	auto getTrimCoil = [=](unsigned int i_coil) -> DataRef<Filament>::Client {
		if(reader.isCoilsDBSet())
			return getCoil(reader.getCoilsDBSet().getTrimCoilIDs()[i_coil]);
		else {
			KJ_REQUIRE(reader.isCustomCoilSet());
			return reader.getCustomCoilSet().getTrimCoils()[i_coil];
		}
	};
	
	auto getControlCoil = [=](unsigned int i_coil) -> DataRef<Filament>::Client {
		if(reader.isCoilsDBSet())
			return getCoil(reader.getCoilsDBSet().getControlCoilOffset() + i_coil);
		else {
			KJ_REQUIRE(reader.isCustomCoilSet());
			return reader.getCustomCoilSet().getControlCoils()[i_coil];
		}
	};
	
	auto initField = [=](MagneticField::Builder out, DataRef<Filament>::Client coil) {
		auto filField = out.initFilamentField();
		filField.setCurrent(1);
		filField.setWidth(reader.getWidth());
		filField.setWindingNo(MAIN_COIL_WINDINGS);
		filField.initFilament().setRef(coil);
	};
	
	auto publish = [this](MagneticField::Builder b) -> LocalDataRef<MagneticField> {
		return lt -> dataService().publish(lt -> randomID(), b);
	};
			
	auto mainCoils = coilPack.initMainCoils(N_MAIN_COILS);
	for(unsigned int i_coil = 0; i_coil < N_MAIN_COILS; ++i_coil) {
		Temporary<MagneticField> output;
		auto sum = output.initSum(N_MODULES);
		for(unsigned int i_mod = 0; i_mod < N_MODULES; ++i_mod) {
			initField(sum[i_mod], getMainCoil(i_mod, i_coil));
		}
		mainCoils[i_coil] = publish(output);
	}
	
	auto trimCoils = coilPack.initTrimCoils(N_TRIM_COILS);
	for(unsigned int i_coil = 0; i_coil < N_TRIM_COILS; ++i_coil) {
		Temporary<MagneticField> output;
		initField(output, getTrimCoil(i_coil));
		trimCoils[i_coil] = publish(output);
	}
	
	auto controlCoils = coilPack.initControlCoils(N_CONTROL_COILS);
	for(unsigned int i_coil = 0; i_coil < N_CONTROL_COILS; ++i_coil) {
		Temporary<MagneticField> output;
		initField(output, getControlCoil(i_coil));
		controlCoils[i_coil] = publish(output);
	}		
}

// CoilsDB implementation that downloads the coil from a remote location
// Does not perform any local caching.
struct CoilsDBWebservice : public CoilsDB::Server {
	kj::String address;
	LibraryThread lt;
		
	Own<kj::HttpHeaderTable> headerTbl = kj::heap<kj::HttpHeaderTable>();
	
	CoilsDBWebservice(kj::StringPtr address, LibraryThread& lt) :
		address(kj::heapString(address)),
		lt(lt -> addRef())
	{}
	
	Promise<void> getCoil(GetCoilContext context)  {
		using kj::HttpHeaderTable;
		using kj::HttpHeaders;
		using capnp::JsonCodec;
		
		auto client = kj::newHttpClient(
			lt->timer(),
			*headerTbl,
			lt -> network(),
			nullptr
		);
		auto response = client->request(
			kj::HttpMethod::GET,
			kj::str(address, "/coil/", context.getParams().getId(), "/data"),
			HttpHeaders(*headerTbl)
		).response;
		
		auto read = response.then([](auto response) { KJ_REQUIRE(response.statusCode == 200); return response.body->readAllText().attach(mv(response.body)); });
		return read.then([context](kj::String rawJson) mutable {			
			Temporary<CoilsDBCoil> tmp;
			JsonCodec().decode(rawJson, tmp);
			
			auto plf = tmp.getPolylineFilament();
			auto x1 = plf.getX1();
			auto x2 = plf.getX2();
			auto x3 = plf.getX3();
			
			KJ_REQUIRE(x1.size() == x2.size());
			KJ_REQUIRE(x1.size() == x3.size());
			
			auto output = context.getResults().initFilament().initInline();
			output.setShape({x1.size(), 3});
			auto data = output.initData(3 * x1.size());
			
			for(unsigned int i = 0; i < x1.size(); ++i) {
				data.set(3 * i + 0, x1[i]);
				data.set(3 * i + 1, x2[i]);
				data.set(3 * i + 2, x3[i]);
			}
		}).attach(mv(response), mv(client), thisCap());	
	}
	
	Promise<void> getConfig(GetConfigContext context) {
		using kj::HttpHeaderTable;
		using kj::HttpHeaders;
		using capnp::JsonCodec;
		
		auto client = kj::newHttpClient(
			lt->timer(),
			*headerTbl,
			lt -> network(),
			nullptr
		);
		auto response = client->request(
			kj::HttpMethod::GET,
			kj::str(address, "/config/", context.getParams().getId(), "/data"),
			HttpHeaders(*headerTbl)
		).response;
		
		auto read = response.then([](auto response) { KJ_REQUIRE(response.statusCode == 200); return response.body->readAllText().attach(mv(response.body)); });
		return read.then([context](kj::String rawJson) mutable {			
			Temporary<CoilsDBConfig> tmp;
			JsonCodec().decode(rawJson, context.getResults().initConfig());
		}).attach(mv(headerTbl), mv(response), mv(client), thisCap());	
	}
};

struct OfflineCoilsDB : public CoilsDB::Server {
	LibraryThread lt;
	Array<LocalDataRef<OfflineData>> offlineData;
	
	CoilsDB::Client backend;
	
	OfflineCoilsDB(ArrayPtr<LocalDataRef<OfflineData>> offlineData, LibraryThread& lt, CoilsDB::Client backend) :
		lt(lt->addRef()),
		offlineData(kj::heapArray(offlineData)),
		backend(backend)
	{}
	
	Promise<void> getCoil(GetCoilContext context) {
		for(auto& entry : offlineData) {
			for(auto coilEntry : entry.get().getW7xCoils()) {
				if(coilEntry.getId() == context.getParams().getId()) {
					context.getResults().setFilament(coilEntry.getFilament());
					return kj::READY_NOW;
				}
			}
		}
		
		auto tail = backend.getCoilRequest();
		tail.setId(context.getParams().getId());
		return context.tailCall(mv(tail));
	}
	
	Promise<void> getConfig(GetConfigContext context) {
		for(auto& entry : offlineData) {
			for(auto configEntry : entry.get().getW7xConfigs()) {
				if(configEntry.getId() == context.getParams().getId()) {
					context.getResults().setConfig(configEntry.getConfig());
					return kj::READY_NOW;
				}
			}
		}
		
		auto tail = backend.getConfigRequest();
		tail.setId(context.getParams().getId());
		return context.tailCall(mv(tail));
	}
};

CoilsDB::Client newCoilsDBFromWebservice(kj::StringPtr address, LibraryThread& lt) {
	return CoilsDB::Client(kj::heap<CoilsDBWebservice>(address, lt -> addRef()));
}

CoilsDB::Client newOfflineCoilsDB(ArrayPtr<LocalDataRef<OfflineData>> offlineData, LibraryThread& lt, CoilsDB::Client backend) {
	return CoilsDB::Client(kj::heap<OfflineCoilsDB>(offlineData, lt, backend));
}
	
}}

}