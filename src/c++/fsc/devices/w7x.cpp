#include <kj/compat/http.h>
#include <capnp/compat/json.h>

#include "w7x.h"

namespace fsc {

namespace devices { namespace w7x {

// === class CoilsDBResolver ===

constexpr unsigned int CoilsDBResolver::N_MAIN_COILS;
constexpr unsigned int CoilsDBResolver::N_MODULES ;
constexpr unsigned int CoilsDBResolver::N_TRIM_COILS;
constexpr unsigned int CoilsDBResolver::N_CONTROL_COILS ;
	
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
					
					return coilsDBResponse.then([this, output, cdbConfig](auto config) mutable {						
						auto coils = config.getCoils();
						auto currents = config.getCurrents();
						
						double scale = config.getScale();
						
						unsigned int n_coils = coils.size();
						KJ_REQUIRE(currents.size() == n_coils);
						
						auto sum = output.initSum(n_coils);
						for(unsigned int i = 0; i < n_coils; ++i) {
							auto filField = sum[i].initFilamentField();
							filField.setCurrent(scale * currents[i]);
							filField.setBiotSavartSettings(cdbConfig.getBiotSavartSettings());
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
	
	return getCoilFields(reader.getCoils()).then([reader, output, context, this](LocalDataRef<CoilFields> coilFields) mutable {
		CoilFields::Reader fieldsReader = coilFields.get();
		
		using FieldRef = DataRef<MagneticField>::Client;
		unsigned int offset = 0;
		
		auto addOutput = [&](FieldRef field, double current) mutable {
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
		
		double controlCurrents[10];
		auto cc = reader.getControl();
		for(unsigned int i = 0; i < 10; ++i) {
			switch(cc.size()) {
				case 10:
					controlCurrents[i] = cc[i];
					break;
				case 2:
					controlCurrents[i] = cc[i % 2];
					break;
				case 5:
					// TODO: Check if the control coils are grouped by module
					controlCurrents[i] = cc[i / 2];
					break;
				case 0:
					controlCurrents[i] = 0;
				default:
					KJ_FAIL_REQUIRE("Control coil currents must be of length 0 (no CC), 2 (upper and lower), 5 (one per module), or 10 (upper and lower per module)", cc.size());
			}
		}
		
		for(unsigned int i = 0; i < N_CONTROL_COILS; ++i) {
			addOutput(fieldsReader.getControlCoils()[i], controlCurrents[i]);
		}
		
		KJ_ASSERT(offset == output.getSum().size());
	});
}

Promise<LocalDataRef<CoilFields>> CoilsDBResolver::getCoilFields(W7XCoilSet::Reader reader) {	
	return ID::fromReaderWithRefs(reader)
	.then([this, reader](ID id) {
		KJ_IF_MAYBE(coilPack, coilPacks.find(id)) {
			return *coilPack;
		}
		
		capnp::MallocMessageBuilder tmp;
		auto coilPack = tmp.initRoot<CoilFields>();
		buildCoilFields(reader, coilPack);
		
		auto ref = lt->dataService().publish(lt->randomID(), coilPack);
		coilPacks.insert(id, ref);
		return ref;
	});
}

Promise<void> CoilsDBResolver::processFilament(Filament::Reader input, Filament::Builder output, ResolveContext context) {
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
	
	DataRef<Filament>::Client newCoil = coilRequest.send().then([cdbID, this](auto filament) {
		// auto filament = response.getFilament();
		auto ref = lt -> dataService().publish(lt -> randomID(), Filament::Reader(filament));
		
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
	
	auto initField = [=](MagneticField::Builder out, DataRef<Filament>::Client coil, unsigned int windingNo, bool invert) {
		auto filField = out.initFilamentField();
		filField.setCurrent(invert ? 1 : -1);
		filField.setBiotSavartSettings(reader.getBiotSavartSettings());
		filField.setWindingNo(windingNo);
		filField.initFilament().setRef(coil);
	};
	
	auto publish = [this](MagneticField::Builder b) -> LocalDataRef<MagneticField> {
		return lt -> dataService().publish(lt -> randomID(), b);
	};
			
	auto mainCoils = coilPack.initMainCoils(N_MAIN_COILS);
	auto nWindMain = reader.getNWindMain();
	for(unsigned int i_coil = 0; i_coil < N_MAIN_COILS; ++i_coil) {
		Temporary<MagneticField> output;
		auto sum = output.initSum(N_MODULES);
		for(unsigned int i_mod = 0; i_mod < N_MODULES; ++i_mod) {
			initField(sum[i_mod], getMainCoil(i_mod, i_coil), nWindMain[i_coil], reader.getInvertMainCoils());
		}
		mainCoils[i_coil] = publish(output);
	}
	
	auto trimCoils = coilPack.initTrimCoils(N_TRIM_COILS);
	auto nWindTrim = reader.getNWindTrim();
	for(unsigned int i_coil = 0; i_coil < N_TRIM_COILS; ++i_coil) {
		Temporary<MagneticField> output;
		initField(output, getTrimCoil(i_coil), nWindTrim[i_coil], false);
		trimCoils[i_coil] = publish(output);
	}
	
	auto controlCoils = coilPack.initControlCoils(N_CONTROL_COILS);
	auto nWindControl = reader.getNWindControl();
	for(unsigned int i_coil = 0; i_coil < N_CONTROL_COILS; ++i_coil) {
		Temporary<MagneticField> output;
		initField(output, getControlCoil(i_coil), nWindControl[i_coil], reader.getInvertControlCoils()[i_coil]);
		controlCoils[i_coil] = publish(output);
	}		
}

// === class ComponentsDBResolver ===

constexpr kj::StringPtr ComponentsDBResolver::CDB_ID_TAG;
constexpr kj::StringPtr ComponentsDBResolver::CDB_ASID_TAG;

ComponentsDBResolver::ComponentsDBResolver(LibraryThread& lt, ComponentsDB::Client backend) :
	GeometryResolverBase(lt),
	backend(backend)
{}

Promise<void> ComponentsDBResolver::processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveContext context) {
	return GeometryResolverBase::processGeometry(input, output, context)
	.then([input, output, context, this]() mutable -> Promise<void> {
		switch(input.which()) {
			case Geometry::COMPONENTS_D_B_MESHES: {
				auto ids = input.getComponentsDBMeshes();
				auto n = ids.size();
				auto nodes = output.initCombined(ids.size());
				
				for(decltype(n) i = 0; i < n; ++i) {
					auto node = nodes[i];
					node.setMesh(getComponent(ids[i]));
					
					auto tags = node.initTags(1);
					tags[0].setName(CDB_ID_TAG);
					tags[0].initValue().setUInt64(ids[i]);
				}
				return READY_NOW;
			}
			
			case Geometry::COMPONENTS_D_B_ASSEMBLIES: {
				auto ids = input.getComponentsDBAssemblies();
				auto n = ids.size();
				auto nodes = output.initCombined(ids.size());
				
				auto subTasks = kj::heapArrayBuilder<Promise<void>>(n);
				for(decltype(n) i = 0; i < n; ++i) {
					auto node = nodes[i];
					
					auto tags = node.initTags(1);
					tags[0].setName(CDB_ASID_TAG);
					tags[0].initValue().setUInt64(ids[i]);
					
					subTasks.add(
						getAssembly(ids[i])
						.then([=](kj::Array<uint64_t> cids) mutable {
							Temporary<Geometry> intermediate;
							auto tmpIds = intermediate.initComponentsDBMeshes(cids.size());
							for(unsigned int i = 0; i < cids.size(); ++i)
								tmpIds.set(i, cids[i]);
							
							return processGeometry(intermediate, node, context).attach(mv(intermediate));
						})
					);
				}
				return kj::joinPromises(subTasks.finish());
			}
			
			default:
				return READY_NOW;
		}
	});
}

Promise<kj::Array<uint64_t>> ComponentsDBResolver::getAssembly(uint64_t id) {
	auto request = backend.getAssemblyRequest();
	request.setId(id);
	
	return request.send()
	.then([](auto response) {
		auto components = response.getComponents();
		unsigned int n = components.size();
		
		auto result = kj::heapArray<uint64_t>(n);
		for(unsigned int i = 0; i < n; ++i)
			result[i] = components[i];

		return mv(result);
	});
}

DataRef<Mesh>::Client ComponentsDBResolver::getComponent(uint64_t id) {
	KJ_IF_MAYBE(pMesh, meshes.find(id)) {
		return *pMesh;
	}
	
	auto request = backend.getMeshRequest();
	request.setId(id);
	
	return request.send()
	.then([id, this](capnp::Response<Mesh> response) -> DataRef<Mesh>::Client {
		KJ_IF_MAYBE(pMesh, meshes.find(id)) {
			return *pMesh;
		}
		
		DataRef<Mesh>::Client published = lt->dataService().publish(lt->randomID(), (Mesh::Reader&) response);
		meshes.insert(id, published);
		return published;
	});
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
	
	Promise<void> getCoil(GetCoilContext context) override {
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
			
			auto output = context.initResults().initInline();
			output.setShape({x1.size(), 3});
			auto data = output.initData(3 * x1.size());
			
			for(unsigned int i = 0; i < x1.size(); ++i) {
				data.set(3 * i + 0, x1[i]);
				data.set(3 * i + 1, x2[i]);
				data.set(3 * i + 2, x3[i]);
			}
		}).attach(mv(response), mv(client), thisCap());	
	}
	
	Promise<void> getConfig(GetConfigContext context) override {
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
			JsonCodec().decode(rawJson, context.initResults());
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
	
	Promise<void> getCoil(GetCoilContext context) override {
		for(auto& entry : offlineData) {
			for(auto coilEntry : entry.get().getW7xCoils()) {
				if(coilEntry.getId() == context.getParams().getId()) {
					context.setResults(coilEntry.getFilament());
					return kj::READY_NOW;
				}
			}
		}
		
		auto tail = backend.getCoilRequest();
		tail.setId(context.getParams().getId());
		return context.tailCall(mv(tail));
	}
	
	Promise<void> getConfig(GetConfigContext context) override {
		for(auto& entry : offlineData) {
			for(auto configEntry : entry.get().getW7xConfigs()) {
				if(configEntry.getId() == context.getParams().getId()) {
					context.setResults(configEntry.getConfig());
					return kj::READY_NOW;
				}
			}
		}
		
		auto tail = backend.getConfigRequest();
		tail.setId(context.getParams().getId());
		return context.tailCall(mv(tail));
	}
};

struct ComponentsDBWebservice : public ComponentsDB::Server {
	kj::String address;
	LibraryThread lt;
		
	Own<kj::HttpHeaderTable> headerTbl = kj::heap<kj::HttpHeaderTable>();
	
	ComponentsDBWebservice(kj::StringPtr address, LibraryThread& lt) :
		address(kj::heapString(address)),
		lt(lt -> addRef())
	{}
	
	Promise<void> getMesh(GetMeshContext context) override  {
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
			kj::str(address, "/component/", context.getParams().getId(), "/data"),
			HttpHeaders(*headerTbl)
		).response;
		
		auto read = response.then([](auto response) { KJ_REQUIRE(response.statusCode == 200); return response.body->readAllText().attach(mv(response.body)); });
		return read.then([context](kj::String rawJson) mutable {			
			Temporary<ComponentsDBMesh> tmp;
			JsonCodec().decode(rawJson, tmp);
		
			auto inMesh = tmp.getSurfaceMesh();
			
			auto nPoints = inMesh.getNodes().getX1().size();
			KJ_REQUIRE(inMesh.getNodes().getX2().size() == nPoints);
			KJ_REQUIRE(inMesh.getNodes().getX3().size() == nPoints);
			
			// Translate components DB response to native format
			Temporary<Mesh> mesh;
			
			// Step 1: Translate vertices
			auto verts = mesh.initVertices();
			verts.setShape({nPoints, 3});
			auto data = verts.initData(3 * nPoints);
			for(unsigned int i = 0; i < nPoints; ++i) {
				data.set(3 * i + 0, inMesh.getNodes().getX1()[i]);
				data.set(3 * i + 1, inMesh.getNodes().getX2()[i]);
				data.set(3 * i + 2, inMesh.getNodes().getX3()[i]);
			}
			
			// Step 2: Translate indices
			// TODO: I think components db uses 1 based indices? not sure
			
			auto inPoly = inMesh.getPolygons();
			auto outPoly = mesh.initIndices(inPoly.size());
			for(unsigned int i = 0; i < inPoly.size(); ++i) {
				KJ_REQUIRE(inPoly[i] > 0, "Assumption violated that ComponentsDB uses 1-based indices", context.getParams().getId(), i);
				
				outPoly.set(i, inPoly[i] - 1);
			}
			
			// Step 3: Translate polygon sizes
			auto nVert = inMesh.getNumVertices();
			if(nVert.size() == 0) {
				mesh.initPolyMesh(0);
			} else {
				auto firstNVert = nVert[0];
				bool allSame = true;
				
				for(unsigned int i = 0; i < nVert.size(); ++i) {
					auto polySize = nVert[i];
					
					if(polySize != firstNVert)
						allSame = false;
				}
				
				if(firstNVert == 3 && allSame && 3 * nVert.size() == inPoly.size()) {
					// We have a triangle mesh
					// Let's save ourselves the polygon buffer
					mesh.setTriMesh();
				} else {
					auto polyMesh = mesh.initPolyMesh(nVert.size() + 1);
					
					uint32_t offset = 0;
					
					for(unsigned int i = 0; i < nVert.size(); ++i) {
						auto polySize = nVert[i];
						
						polyMesh.set(i, offset);
						offset += polySize;
					}
					polyMesh.set(nVert.size(), offset);
				}
			}
			
			context.setResults(mesh.asReader());
		}).attach(mv(response), mv(client), thisCap());	
	}
	
	Promise<void> getAssembly(GetAssemblyContext context) override {
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
			kj::str(address, "/component/", context.getParams().getId(), "/data"),
			HttpHeaders(*headerTbl)
		).response;
		
		auto read = response.then([](auto response) { KJ_REQUIRE(response.statusCode == 200); return response.body->readAllText().attach(mv(response.body)); });
		return read.then([context](kj::String rawJson) mutable {			
			Temporary<ComponentsDBAssembly> tmp;
			JsonCodec().decode(rawJson, tmp);
			
			context.initResults().setComponents(tmp.getComponents());
		}).attach(mv(response), mv(client), thisCap());	
	}
};

struct OfflineComponentsDB : public ComponentsDB::Server {
	LibraryThread lt;
	Array<LocalDataRef<OfflineData>> offlineData;
	
	ComponentsDB::Client backend;
	
	OfflineComponentsDB(ArrayPtr<LocalDataRef<OfflineData>> offlineData, LibraryThread& lt, ComponentsDB::Client backend) :
		lt(lt->addRef()),
		offlineData(kj::heapArray(offlineData)),
		backend(backend)
	{}
	
	Promise<void> getMesh(GetMeshContext context) override {
		for(auto& entry : offlineData) {
			for(auto componentEntry : entry.get().getW7xComponents()) {
				if(componentEntry.getId() == context.getParams().getId()) {
					context.setResults(componentEntry.getComponent());
					return kj::READY_NOW;
				}
			}
		}
		
		auto tail = backend.getMeshRequest();
		tail.setId(context.getParams().getId());
		return context.tailCall(mv(tail));
	}
	
	Promise<void> getAssembly(GetAssemblyContext context) override {
		for(auto& entry : offlineData) {
			for(auto assemblyEntry : entry.get().getW7xAssemblies()) {
				if(assemblyEntry.getId() == context.getParams().getId()) {
					context.initResults().setComponents(assemblyEntry.getAssembly());
					return kj::READY_NOW;
				}
			}
		}
		
		auto tail = backend.getAssemblyRequest();
		tail.setId(context.getParams().getId());
		return context.tailCall(mv(tail));
	}
};

CoilsDB::Client newCoilsDBFromWebservice(kj::StringPtr address, LibraryThread& lt) {
	return CoilsDB::Client(kj::heap<CoilsDBWebservice>(address, lt));
}

CoilsDB::Client newOfflineCoilsDB(ArrayPtr<LocalDataRef<OfflineData>> offlineData, LibraryThread& lt, CoilsDB::Client backend) {
	return CoilsDB::Client(kj::heap<OfflineCoilsDB>(offlineData, lt, backend));
}

ComponentsDB::Client newComponentsDBFromWebservice(kj::StringPtr address, LibraryThread& lt) {
	return ComponentsDB::Client(kj::heap<ComponentsDBWebservice>(address, lt));
}

ComponentsDB::Client newOfflineComponentsDB(ArrayPtr<LocalDataRef<OfflineData>> offlineData, LibraryThread& lt, ComponentsDB::Client backend) {
	return ComponentsDB::Client(kj::heap<OfflineComponentsDB>(offlineData, lt, backend));
}
	
}}

}
