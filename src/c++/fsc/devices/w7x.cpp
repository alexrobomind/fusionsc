#include <kj/compat/http.h>
#include <capnp/compat/json.h>

#include "w7x.h"

namespace fsc {

namespace devices { namespace w7x {
	
namespace {
	
constexpr static unsigned int N_MAIN_COILS = 7;
constexpr static unsigned int N_MODULES = 10;
constexpr static unsigned int N_TRIM_COILS = 5;
constexpr static unsigned int N_CONTROL_COILS = 10;

struct W7XFieldResolver : public FieldResolverBase {
	Promise<void> processField   (MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context) override;
	
	// Helper function to handle the "coilsAndCurrents" type of W7-X magnetic config
	void coilsAndCurrents(MagneticField::W7x::CoilsAndCurrents::Reader reader, MagneticField::Builder output, ResolveFieldContext context);
};

struct W7XGeometryResolver : public GeometryResolverBase {
	Promise<void> processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveGeometryContext context) override;
};
	
/**
 * Magnetic field resolver that processes CoilsDB references.
 */
struct CoilsDBResolver : public FieldResolverBase {	
	CoilsDBResolver(CoilsDB::Client backend) : backend(mv(backend)) {}
	
	kj::TreeMap<uint64_t, DataRef<Filament>::Client> coils;
	CoilsDB::Client backend;
	
	Promise<void> processFilament(Filament     ::Reader input, Filament     ::Builder output, ResolveFieldContext context) override;
	
	//! Loads the coils from the backend coilsDB. Caches the result
	DataRef<Filament>::Client getCoil(uint64_t cdbID);	
};
	
/**
 * Magnetic field resolver that processes W7-X configuration descriptions
 */
struct ConfigDBResolver : public FieldResolverBase {
	ConfigDBResolver(CoilsDB::Client backend) : backend(mv(backend)) {}
	
	CoilsDB::Client backend;
	
	Promise<void> processField (MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context) override;
};

struct ComponentsDBResolver : public GeometryResolverBase {
	constexpr static kj::StringPtr CDB_ID_TAG = "w7x-component-id"_kj;
	constexpr static kj::StringPtr CDB_ASID_TAG = "w7x-assembly-id"_kj;
	
	ComponentsDBResolver(ComponentsDB::Client backend);
	
	kj::TreeMap<uint64_t, DataRef<Mesh>::Client> meshes;
	ComponentsDB::Client backend;
	
	Promise<void> processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveGeometryContext context) override;
	
	// Loads a component from the components db (caches result)
	DataRef<Mesh>::Client getComponent(uint64_t cdbID);
	
	// Loads an assembly from the components db
	Promise<Array<uint64_t>> getAssembly(uint64_t cdbID);
};

// === class W7XFieldResolver ===

Promise<void> W7XFieldResolver::processField(MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context) {
	if(!input.isW7x())
		return FieldResolverBase::processField(input, output, context);
	
	auto w7x = input.getW7x();
	
	if(!w7x.isCoilsAndCurrents()) {
		output.setNested(input);
		return READY_NOW;
	}
	
	coilsAndCurrents(w7x.getCoilsAndCurrents(), output, context);
	return READY_NOW;
}

// === class W7XGeometryResolver ===

Promise<void> W7XGeometryResolver::processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveGeometryContext context) {
	if(!input.isW7x())
		return GeometryResolverBase::processGeometry(input, output, context);
	
	using W7X = Geometry::W7x;
	W7X::Reader w7x = input.getW7x();
	
	switch(w7x.which()) {
		case W7X::OP21_DIVERTOR: {
			auto ref = getActiveThread().dataService().publishConstant<Geometry>(W7X_OP21_DIVERTOR.get());
			auto geometry = ref.get();
			output.setNested(geometry);
			output.setTags(input.getTags());
			return READY_NOW;
		}
		case W7X::OP21_BAFFLES: {
			auto ref = getActiveThread().dataService().publishConstant<Geometry>(W7X_OP21_BAFFLES_NO_HOLES.get());
			auto geometry = ref.get();
			output.setNested(geometry);
			output.setTags(input.getTags());
			return READY_NOW;
		}
		case W7X::OP21_HEAT_SHIELD: {
			auto ref = getActiveThread().dataService().publishConstant<Geometry>(W7X_OP21_HEAT_SHIELD_NO_HOLES.get());
			auto geometry = ref.get();
			output.setNested(geometry);
			output.setTags(input.getTags());
			return READY_NOW;
		}
		default:
			break;
	}
	
	output.setNested(input);
	return READY_NOW;
}

// === class CoilsDBResolver ===

Promise<void> CoilsDBResolver::processFilament(Filament::Reader input, Filament::Builder output, ResolveFieldContext context) {
	if(input.which() != Filament::W7X)
		return FieldResolverBase::processFilament(input, output, context);
	
	auto w7x = input.getW7x();
	uint64_t coilId = w7x.getCoilsDb();
	auto coil = getCoil(coilId);
	
	output.setRef(coil);
	return coil.whenResolved();
}

DataRef<Filament>::Client CoilsDBResolver::getCoil(uint64_t cdbID) {
	KJ_IF_MAYBE(coilsRef, coils.findEntry(cdbID)) {
		return coilsRef->value;
	}
	 
	auto coilRequest = backend.getCoilRequest();
	coilRequest.setId(cdbID);
	
	DataRef<Filament>::Client newCoil = coilRequest.send()
	.then([cdbID, this](auto filament) -> DataRef<Filament>::Client {
		auto ref = getActiveThread().dataService().publish(Filament::Reader(filament));
		
		KJ_IF_MAYBE(coilsRef, coils.findEntry(cdbID)) {
			return coilsRef->value;
		}
			
		coils.insert(cdbID, ref);
		
		return ref;
	})
	.attach(thisCap());
	
	return newCoil;
}

// === class ConfigDBResolver ===

Promise<void> ConfigDBResolver::processField(MagneticField::Reader input, MagneticField::Builder output, ResolveFieldContext context) {
	if(input.which() != MagneticField::W7X)
		return FieldResolverBase::processField(input, output, context);
	
	auto w7x = input.getW7x();
	
	if(!w7x.isConfigurationDb()) {
		output.setNested(input);
		return READY_NOW;
	}
	
	auto cdbConfig = w7x.getConfigurationDb();
	
	auto request = backend.getConfigRequest();
	request.setId(cdbConfig.getConfigId());
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
			filField.initFilament().initW7x().setCoilsDb(coils[i]);
		}
	});
}

}

void W7XFieldResolver::coilsAndCurrents(MagneticField::W7x::CoilsAndCurrents::Reader reader, MagneticField::Builder output, ResolveFieldContext context) {
	output.initSum(N_MAIN_COILS + N_TRIM_COILS + N_CONTROL_COILS);
	
	// Transform coil fields into actual fields
	Temporary<W7XCoilSet> builtSet;
	auto fields = builtSet.initFields();
	buildCoilFields(reader.getCoils(), fields);
	
	unsigned int offset = 0;
	
	auto addOutput = [&](MagneticField::Builder field, double current) mutable {
		auto subField = output.getSum()[offset++];
		auto scaleBy = subField.initScaleBy();
		scaleBy.setFactor(current);
		scaleBy.setField(field);			
	};
	
	for(unsigned int i = 0; i < 5; ++i) {
		addOutput(fields.getMainFields()[i], reader.getNonplanar()[i]);
	}
	
	for(unsigned int i = 0; i < 2; ++i) {
		addOutput(fields.getMainFields()[i + 5], reader.getPlanar()[i]);
	}
	
	KJ_ASSERT(offset == N_MAIN_COILS);
	
	for(unsigned int i = 0; i < N_TRIM_COILS; ++i) {
		addOutput(fields.getTrimFields()[i], reader.getTrim()[i]);
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
		addOutput(fields.getControlFields()[i], controlCurrents[i]);
	}
	
	KJ_ASSERT(offset == output.getSum().size());
}

// === class ComponentsDBResolver ===

constexpr kj::StringPtr ComponentsDBResolver::CDB_ID_TAG;
constexpr kj::StringPtr ComponentsDBResolver::CDB_ASID_TAG;

ComponentsDBResolver::ComponentsDBResolver(ComponentsDB::Client backend) :
	backend(backend)
{}

Promise<void> ComponentsDBResolver::processGeometry(Geometry::Reader input, Geometry::Builder output, ResolveGeometryContext context) {
	return GeometryResolverBase::processGeometry(input, output, context)
	.then([input, output, context, this]() mutable -> Promise<void> {
		if(input.which() != Geometry::W7X)
			return READY_NOW;
		
		auto w7x = input.getW7x();
		switch(w7x.which()) {
			case Geometry::W7x::COMPONENTS_DB_MESH: {
				auto id = w7x.getComponentsDbMesh();
				output.setMesh(getComponent(id));
					
				auto tags = output.initTags(1);
				tags[0].setName(CDB_ID_TAG);
				tags[0].initValue().setUInt64(id);
				
				return output.getMesh().whenResolved();
			}
			
			case Geometry::W7x::COMPONENTS_DB_ASSEMBLY: {
				auto id = w7x.getComponentsDbAssembly();
				auto tags = output.initTags(1);
				tags[0].setName(CDB_ASID_TAG);
				tags[0].initValue().setUInt64(id);
				
				return getAssembly(id)
				.then([this, output](kj::Array<uint64_t> cids) mutable {
					auto nodes = output.initCombined(cids.size());
					
					auto promises = kj::heapArrayBuilder<Promise<void>>(cids.size());
					
					for(auto i : kj::indices(cids)) {
						auto node = nodes[i];
						auto id = cids[i];
						
						node.setMesh(getComponent(id));
							
						auto tags = node.initTags(1);
						tags[0].setName(CDB_ID_TAG);
						tags[0].initValue().setUInt64(id);
						
						promises.add(node.getMesh().whenResolved());
					}
					
					return kj::joinPromises(promises.finish());
				});
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
		
		DataRef<Mesh>::Client published = getActiveThread().dataService().publish((Mesh::Reader&) response);
		meshes.insert(id, published);
		
		return published;
	}).attach(thisCap());
}

// CoilsDB implementation that downloads the coil from a remote location
// Does not perform any local caching.
struct CoilsDBWebservice : public CoilsDB::Server {
	kj::String address;
		
	Own<kj::HttpHeaderTable> headerTbl = kj::heap<kj::HttpHeaderTable>();
	
	CoilsDBWebservice(kj::StringPtr address) :
		address(kj::heapString(address))
	{}
	
	Promise<void> getCoil(GetCoilContext context) override {
		using kj::HttpHeaderTable;
		using kj::HttpHeaders;
		using capnp::JsonCodec;
		
		auto client = kj::newHttpClient(
			getActiveThread().timer(),
			*headerTbl,
			getActiveThread().network(),
			nullptr
		);
		auto response = client->request(
			kj::HttpMethod::GET,
			kj::str(address, "/coil/", context.getParams().getId(), "/data"),
			HttpHeaders(*headerTbl)
		).response;
		
		auto read = response.then([](auto response) { KJ_REQUIRE(response.statusCode == 200); return response.body->readAllText().attach(mv(response.body)); });
		auto result = read.then([context](kj::String rawJson) mutable {			
			Temporary<CoilsDBCoil> tmp;
			// JsonCodec().decode(rawJson, tmp);
			JsonCodec codec;
			codec.setRejectUnknownFields(true);
			codec.decode(rawJson, tmp);
			
			auto plf = tmp.getPolylineFilament().getVertices();
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
		
		return getActiveThread().timer().timeoutAfter(1 * kj::MINUTES, mv(result));	
	}
	
	Promise<void> getConfig(GetConfigContext context) override {
		using kj::HttpHeaderTable;
		using kj::HttpHeaders;
		using capnp::JsonCodec;
		
		auto client = kj::newHttpClient(
			getActiveThread().timer(),
			*headerTbl,
			getActiveThread().network(),
			nullptr
		);
		auto response = client->request(
			kj::HttpMethod::GET,
			kj::str(address, "/config/", context.getParams().getId(), "/data"),
			HttpHeaders(*headerTbl)
		).response;
		
		auto read = response.then([](auto response) { KJ_REQUIRE(response.statusCode == 200); return response.body->readAllText().attach(mv(response.body)); });
		auto result = read.then([context](kj::String rawJson) mutable {			
			Temporary<CoilsDBConfig> tmp;
			JsonCodec().decode(rawJson, context.initResults());
		}).attach(mv(response), mv(client), thisCap());
		
		return getActiveThread().timer().timeoutAfter(1 * kj::MINUTES, mv(result));
	}
};

struct ComponentsDBWebservice : public ComponentsDB::Server {
	kj::String address;
		
	Own<kj::HttpHeaderTable> headerTbl = kj::heap<kj::HttpHeaderTable>();
	
	ComponentsDBWebservice(kj::StringPtr address) :
		address(kj::heapString(address))
	{}
	
	Promise<void> getMesh(GetMeshContext context) override  {
		using kj::HttpHeaderTable;
		using kj::HttpHeaders;
		using capnp::JsonCodec;
		
		auto client = kj::newHttpClient(
			getActiveThread().timer(),
			*headerTbl,
			getActiveThread().network(),
			nullptr
		);
		auto response = client->request(
			kj::HttpMethod::GET,
			kj::str(address, "/component/", context.getParams().getId(), "/data"),
			HttpHeaders(*headerTbl)
		).response;
		
		auto read = response.then([](auto response) { KJ_REQUIRE(response.statusCode == 200); return response.body->readAllText().attach(mv(response.body)); });
		auto result = read.then([context](kj::String rawJson) mutable {				
			Temporary<ComponentsDBMesh> tmp;
			JsonCodec().decode(rawJson, tmp);
			// YAML::Node asYaml = YAML::Load(rawJson.cStr());
			// load(tmp.asBuilder(), asYaml);
		
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
			
			uint32_t minPolyIdx = kj::maxValue;
			
			auto inPoly = inMesh.getPolygons();
			for(uint32_t e : inPoly)
				minPolyIdx = std::min(minPolyIdx, e);
			
			if(minPolyIdx > 1 && inPoly.size() > 0) {
				KJ_DBG(minPolyIdx <= 1, "Assumption violated that ComponentsDB uses either 0- or 1-based indices in mesh", context.getParams().getId(), minPolyIdx, inPoly.size());
			}
			
			auto outPoly = mesh.initIndices(inPoly.size());
			for(auto i : kj::indices(inPoly)) {				
				outPoly.set(i, inPoly[i] - minPolyIdx);
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
		
		return getActiveThread().timer().timeoutAfter(1 * kj::MINUTES, mv(result));
	}
	
	Promise<void> getAssembly(GetAssemblyContext context) override {
		using kj::HttpHeaderTable;
		using kj::HttpHeaders;
		using capnp::JsonCodec;
		
		auto client = kj::newHttpClient(
			getActiveThread().timer(),
			*headerTbl,
			getActiveThread().network(),
			nullptr
		);
		auto response = client->request(
			kj::HttpMethod::GET,
			kj::str(address, "/assembly/", context.getParams().getId(), "/data"),
			HttpHeaders(*headerTbl)
		).response;
		
		auto read = response.then([](auto response) { KJ_REQUIRE(response.statusCode == 200); return response.body->readAllText().attach(mv(response.body)); });
		auto result = read.then([context](kj::String rawJson) mutable {			
			Temporary<ComponentsDBAssembly> tmp;
			JsonCodec().decode(rawJson, tmp);
			
			context.initResults().setComponents(tmp.getComponents());
		}).attach(mv(response), mv(client), thisCap());
		return getActiveThread().timer().timeoutAfter(1 * kj::MINUTES, mv(result));
	}
};

struct ProviderImpl : public Provider::Server {
	Promise<void> connectCoilsDb(ConnectCoilsDbContext ctx) override {
		ctx.getResults().setService(kj::heap<CoilsDBWebservice>(ctx.getParams().getAddress()));
		return READY_NOW;
	}
	
	Promise<void> connectComponentsDb(ConnectComponentsDbContext ctx) override {
		ctx.getResults().setService(kj::heap<ComponentsDBWebservice>(ctx.getParams().getAddress()));
		return READY_NOW;
	}
};

CoilsDB::Client newCoilsDBFromWebservice(kj::StringPtr address) {
	return kj::heap<CoilsDBWebservice>(address);
}

ComponentsDB::Client newComponentsDBFromWebservice(kj::StringPtr address) {
	return kj::heap<ComponentsDBWebservice>(address);
}

Provider::Client newProvider() {
	return kj::heap<ProviderImpl>();
}

FieldResolver::Client newW7xFieldResolver() {
	return kj::heap<W7XFieldResolver>();
}

GeometryResolver::Client newW7xGeometryResolver() {
	return kj::heap<W7XGeometryResolver>();
}

FieldResolver::Client newCoilsDBResolver(CoilsDB::Client coilsDB) {
	return kj::heap<CoilsDBResolver>(coilsDB);
}

FieldResolver::Client newConfigDBResolver(CoilsDB::Client coilsDB) {
	return kj::heap<CoilsDBResolver>(coilsDB);
}

GeometryResolver::Client newComponentsDBResolver(ComponentsDB::Client componentsDB) {
	return GeometryResolver::Client(kj::heap<ComponentsDBResolver>(componentsDB));
}

void buildCoilFields(W7XCoilSet::Reader in, W7XCoilSet::Fields::Builder output) {
	if(in.isFields()) {
		auto inFields = in.getFields();
		
		output.setMainFields(inFields.getMainFields());
		output.setTrimFields(inFields.getTrimFields());
		output.setControlFields(inFields.getControlFields());
		
		return;
	}
	
	KJ_REQUIRE(in.isCoils());
	
	auto coils = in.getCoils();
	
	auto getMainCoil = [=](unsigned int i_mod, unsigned int i_coil, Filament::Builder out) {
		if(coils.isCoilsDbSet()) {
			unsigned int offset = coils.getCoilsDbSet().getMainCoilOffset();
			unsigned int coilID = i_coil < 5
				? offset + 5 * i_mod + i_coil
				: offset + 2 * i_mod + (i_coil - 5) + 50;
			out.initW7x().setCoilsDb(coilID);
		} else {
			KJ_REQUIRE(coils.isCustomCoilSet());
			out.setRef(coils.getCustomCoilSet().getMainCoils()[N_MAIN_COILS * i_mod + i_coil]);
		}
	};
	
	auto getTrimCoil = [=](unsigned int i_coil, Filament::Builder out) {
		if(coils.isCoilsDbSet())
			out.initW7x().setCoilsDb(coils.getCoilsDbSet().getTrimCoilIDs()[i_coil]);
		else {
			KJ_REQUIRE(coils.isCustomCoilSet());
			out.setRef(coils.getCustomCoilSet().getTrimCoils()[i_coil]);
		}
	};
	
	auto getControlCoil = [=](unsigned int i_coil, Filament::Builder out) {
		if(coils.isCoilsDbSet())
			out.initW7x().setCoilsDb(coils.getCoilsDbSet().getControlCoilOffset() + i_coil);
		else {
			KJ_REQUIRE(coils.isCustomCoilSet());
			out.setRef(coils.getCustomCoilSet().getControlCoils()[i_coil]);
		}
	};
	
	auto initField = [=](MagneticField::Builder out, unsigned int windingNo, bool invert) {
		auto filField = out.initFilamentField();
		filField.setCurrent(invert ? -1 : 1);
		filField.setBiotSavartSettings(coils.getBiotSavartSettings());
		filField.setWindingNo(windingNo);
		
		return filField;
	};			
	
	auto mainFields = output.initMainFields(N_MAIN_COILS);
	auto nWindMain = coils.getNWindMain();
	for(unsigned int i_coil = 0; i_coil < N_MAIN_COILS; ++i_coil) {
		auto sum = mainFields[i_coil].initSum(N_MODULES);
		
		for(unsigned int i_mod = 0; i_mod < N_MODULES; ++i_mod) {
			auto filField = initField(sum[i_mod], nWindMain[i_coil], coils.getInvertMainCoils());
			getMainCoil(i_mod, i_coil, filField.getFilament());
		}
	}
	
	auto trimFields = output.initTrimFields(N_TRIM_COILS);
	auto nWindTrim = coils.getNWindTrim();
	for(unsigned int i_coil = 0; i_coil < N_TRIM_COILS; ++i_coil) {
		auto filField = initField(trimFields[i_coil], nWindTrim[i_coil], false);
		getTrimCoil(i_coil, filField.getFilament());
	}
	
	auto controlFields = output.initControlFields(N_CONTROL_COILS);
	auto nWindControl = coils.getNWindControl();
	for(unsigned int i_coil = 0; i_coil < N_CONTROL_COILS; ++i_coil) {
		auto filField = initField(controlFields[i_coil], nWindControl[i_coil], coils.getInvertControlCoils()[i_coil]);
		getControlCoil(i_coil, filField.getFilament());
	}
}
	
}}

}
