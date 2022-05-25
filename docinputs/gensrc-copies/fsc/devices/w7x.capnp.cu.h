#pragma once 

#include <cupnp/cupnp.h>
#include "devices/w7x.capnp.h"
#include "../data.capnp.cu.h"
#include "../geometry.capnp.cu.h"
#include "../magnetics.capnp.cu.h"

namespace fsc{
namespace devices{
namespace w7x{
namespace cu{

struct CoilsDB{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	CoilsDB(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct ComponentsDB{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	ComponentsDB(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct CoilFields{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline CoilFields(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char MAIN_COILS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> mutateMainCoils();

	inline const cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> getMainCoils() const;
	inline bool nonDefaultMainCoils() const;

	inline static const unsigned char TRIM_COILS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> mutateTrimCoils();

	inline const cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> getTrimCoils() const;
	inline bool nonDefaultTrimCoils() const;

	inline static const unsigned char CONTROL_COILS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> mutateControlCoils();

	inline const cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> getControlCoils() const;
	inline bool nonDefaultControlCoils() const;

}; // struct fsc::devices::w7x::cu::CoilFields

struct ComponentsDBMesh{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline ComponentsDBMesh(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct SurfaceMesh{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline SurfaceMesh(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct Nodes{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Nodes(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char X1_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateX1();

	inline const cupnp::List<double> getX1() const;
	inline bool nonDefaultX1() const;

	inline static const unsigned char X2_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateX2();

	inline const cupnp::List<double> getX2() const;
	inline bool nonDefaultX2() const;

	inline static const unsigned char X3_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateX3();

	inline const cupnp::List<double> getX3() const;
	inline bool nonDefaultX3() const;

}; // struct fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes

	inline fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes getNodes() const;

	inline static const unsigned char POLYGONS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint32_t> mutatePolygons();

	inline const cupnp::List<uint32_t> getPolygons() const;
	inline bool nonDefaultPolygons() const;

	inline static const unsigned char NUM_VERTICES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint32_t> mutateNumVertices();

	inline const cupnp::List<uint32_t> getNumVertices() const;
	inline bool nonDefaultNumVertices() const;

}; // struct fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh

	inline fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh getSurfaceMesh() const;

}; // struct fsc::devices::w7x::cu::ComponentsDBMesh

struct ComponentsDBAssembly{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline ComponentsDBAssembly(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char COMPONENTS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateComponents();

	inline const cupnp::List<uint64_t> getComponents() const;
	inline bool nonDefaultComponents() const;

}; // struct fsc::devices::w7x::cu::ComponentsDBAssembly

struct CoilsDBCoil{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline CoilsDBCoil(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct PLFilament{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline PLFilament(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char X1_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateX1();

	inline const cupnp::List<double> getX1() const;
	inline bool nonDefaultX1() const;

	inline static const unsigned char X2_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateX2();

	inline const cupnp::List<double> getX2() const;
	inline bool nonDefaultX2() const;

	inline static const unsigned char X3_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateX3();

	inline const cupnp::List<double> getX3() const;
	inline bool nonDefaultX3() const;

}; // struct fsc::devices::w7x::cu::CoilsDBCoil::PLFilament

	inline static const unsigned char POLYLINE_FILAMENT_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::devices::w7x::cu::CoilsDBCoil::PLFilament mutatePolylineFilament();

	inline const fsc::devices::w7x::cu::CoilsDBCoil::PLFilament getPolylineFilament() const;
	inline bool nonDefaultPolylineFilament() const;

}; // struct fsc::devices::w7x::cu::CoilsDBCoil

struct CoilsDBCoilInfo{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline CoilsDBCoilInfo(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct HistoryEntry{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline HistoryEntry(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline uint64_t getTimeStamp() const;
	inline void setTimeStamp(uint64_t newVal);

	inline static const unsigned char AUTHOR_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateAuthor();

	inline const cupnp::Text getAuthor() const;
	inline bool nonDefaultAuthor() const;

	inline static const unsigned char METHOD_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateMethod();

	inline const cupnp::Text getMethod() const;
	inline bool nonDefaultMethod() const;

	inline static const unsigned char COMMENT_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateComment();

	inline const cupnp::Text getComment() const;
	inline bool nonDefaultComment() const;

}; // struct fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry

	inline static const unsigned char NAME_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateName();

	inline const cupnp::Text getName() const;
	inline bool nonDefaultName() const;

	inline static const unsigned char MACHINE_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateMachine();

	inline const cupnp::Text getMachine() const;
	inline bool nonDefaultMachine() const;

	inline static const unsigned char STATE_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateState();

	inline const cupnp::Text getState() const;
	inline bool nonDefaultState() const;

	inline int64_t getQuality() const;
	inline void setQuality(int64_t newVal);

	inline static const unsigned char AUTHOR_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateAuthor();

	inline const cupnp::Text getAuthor() const;
	inline bool nonDefaultAuthor() const;

	inline static const unsigned char LOCATION_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateLocation();

	inline const cupnp::Text getLocation() const;
	inline bool nonDefaultLocation() const;

	inline static const unsigned char ID_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateId();

	inline const cupnp::Text getId() const;
	inline bool nonDefaultId() const;

	inline static const unsigned char METHOD_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateMethod();

	inline const cupnp::Text getMethod() const;
	inline bool nonDefaultMethod() const;

	inline static const unsigned char COMMENT_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateComment();

	inline const cupnp::Text getComment() const;
	inline bool nonDefaultComment() const;

}; // struct fsc::devices::w7x::cu::CoilsDBCoilInfo

struct CoilsDBConfig{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline CoilsDBConfig(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char COILS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateCoils();

	inline const cupnp::List<uint64_t> getCoils() const;
	inline bool nonDefaultCoils() const;

	inline static const unsigned char CURRENTS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateCurrents();

	inline const cupnp::List<double> getCurrents() const;
	inline bool nonDefaultCurrents() const;

	inline double getScale() const;
	inline void setScale(double newVal);

}; // struct fsc::devices::w7x::cu::CoilsDBConfig

struct CoilsDBClient{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	CoilsDBClient(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};


}}}}
// ===== struct fsc::devices::w7x::cu::CoilFields =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::CoilFields> { using Type = fsc::devices::w7x::cu::CoilFields; }; 
} // namespace ::cupnp

cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> fsc::devices::w7x::cu::CoilFields::mutateMainCoils() { 
	CUPNP_REQUIRE(nonDefaultMainCoils());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>>, 0>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilFields::nonDefaultMainCoils() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> fsc::devices::w7x::cu::CoilFields::getMainCoils() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>>, 0>(structure, data, reinterpret_cast<const capnp::word*>(MAIN_COILS_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> fsc::devices::w7x::cu::CoilFields::mutateTrimCoils() { 
	CUPNP_REQUIRE(nonDefaultTrimCoils());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>>, 1>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilFields::nonDefaultTrimCoils() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> fsc::devices::w7x::cu::CoilFields::getTrimCoils() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>>, 1>(structure, data, reinterpret_cast<const capnp::word*>(TRIM_COILS_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> fsc::devices::w7x::cu::CoilFields::mutateControlCoils() { 
	CUPNP_REQUIRE(nonDefaultControlCoils());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>>, 2>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilFields::nonDefaultControlCoils() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>> fsc::devices::w7x::cu::CoilFields::getControlCoils() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::DataRef<fsc::cu::MagneticField>>, 2>(structure, data, reinterpret_cast<const capnp::word*>(CONTROL_COILS_DEFAULT_VALUE));
} 

// ===== struct fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::ComponentsDBMesh::SurfaceMesh::Nodes> { using Type = fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes; }; 
} // namespace ::cupnp

cupnp::List<double> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes::mutateX1() { 
	CUPNP_REQUIRE(nonDefaultX1());
	return cupnp::mutatePointerField<cupnp::List<double>, 0>(structure, data);
} 

bool fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes::nonDefaultX1() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<double> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes::getX1() const { 
	return cupnp::getPointerField<cupnp::List<double>, 0>(structure, data, reinterpret_cast<const capnp::word*>(X1_DEFAULT_VALUE));
} 

cupnp::List<double> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes::mutateX2() { 
	CUPNP_REQUIRE(nonDefaultX2());
	return cupnp::mutatePointerField<cupnp::List<double>, 1>(structure, data);
} 

bool fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes::nonDefaultX2() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<double> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes::getX2() const { 
	return cupnp::getPointerField<cupnp::List<double>, 1>(structure, data, reinterpret_cast<const capnp::word*>(X2_DEFAULT_VALUE));
} 

cupnp::List<double> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes::mutateX3() { 
	CUPNP_REQUIRE(nonDefaultX3());
	return cupnp::mutatePointerField<cupnp::List<double>, 2>(structure, data);
} 

bool fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes::nonDefaultX3() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::List<double> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes::getX3() const { 
	return cupnp::getPointerField<cupnp::List<double>, 2>(structure, data, reinterpret_cast<const capnp::word*>(X3_DEFAULT_VALUE));
} 

// ===== struct fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::ComponentsDBMesh::SurfaceMesh> { using Type = fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh; }; 
} // namespace ::cupnp

fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::getNodes() const { 
	return fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::Nodes(structure, data);
} 

cupnp::List<uint32_t> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::mutatePolygons() { 
	CUPNP_REQUIRE(nonDefaultPolygons());
	return cupnp::mutatePointerField<cupnp::List<uint32_t>, 3>(structure, data);
} 

bool fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::nonDefaultPolygons() const { 
	return cupnp::hasPointerField<3>(structure, data);
} 

const cupnp::List<uint32_t> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::getPolygons() const { 
	return cupnp::getPointerField<cupnp::List<uint32_t>, 3>(structure, data, reinterpret_cast<const capnp::word*>(POLYGONS_DEFAULT_VALUE));
} 

cupnp::List<uint32_t> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::mutateNumVertices() { 
	CUPNP_REQUIRE(nonDefaultNumVertices());
	return cupnp::mutatePointerField<cupnp::List<uint32_t>, 4>(structure, data);
} 

bool fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::nonDefaultNumVertices() const { 
	return cupnp::hasPointerField<4>(structure, data);
} 

const cupnp::List<uint32_t> fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh::getNumVertices() const { 
	return cupnp::getPointerField<cupnp::List<uint32_t>, 4>(structure, data, reinterpret_cast<const capnp::word*>(NUM_VERTICES_DEFAULT_VALUE));
} 

// ===== struct fsc::devices::w7x::cu::ComponentsDBMesh =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::ComponentsDBMesh> { using Type = fsc::devices::w7x::cu::ComponentsDBMesh; }; 
} // namespace ::cupnp

fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh fsc::devices::w7x::cu::ComponentsDBMesh::getSurfaceMesh() const { 
	return fsc::devices::w7x::cu::ComponentsDBMesh::SurfaceMesh(structure, data);
} 

// ===== struct fsc::devices::w7x::cu::ComponentsDBAssembly =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::ComponentsDBAssembly> { using Type = fsc::devices::w7x::cu::ComponentsDBAssembly; }; 
} // namespace ::cupnp

cupnp::List<uint64_t> fsc::devices::w7x::cu::ComponentsDBAssembly::mutateComponents() { 
	CUPNP_REQUIRE(nonDefaultComponents());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

bool fsc::devices::w7x::cu::ComponentsDBAssembly::nonDefaultComponents() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<uint64_t> fsc::devices::w7x::cu::ComponentsDBAssembly::getComponents() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(COMPONENTS_DEFAULT_VALUE));
} 

// ===== struct fsc::devices::w7x::cu::CoilsDBCoil::PLFilament =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::CoilsDBCoil::PLFilament> { using Type = fsc::devices::w7x::cu::CoilsDBCoil::PLFilament; }; 
} // namespace ::cupnp

cupnp::List<double> fsc::devices::w7x::cu::CoilsDBCoil::PLFilament::mutateX1() { 
	CUPNP_REQUIRE(nonDefaultX1());
	return cupnp::mutatePointerField<cupnp::List<double>, 0>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoil::PLFilament::nonDefaultX1() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<double> fsc::devices::w7x::cu::CoilsDBCoil::PLFilament::getX1() const { 
	return cupnp::getPointerField<cupnp::List<double>, 0>(structure, data, reinterpret_cast<const capnp::word*>(X1_DEFAULT_VALUE));
} 

cupnp::List<double> fsc::devices::w7x::cu::CoilsDBCoil::PLFilament::mutateX2() { 
	CUPNP_REQUIRE(nonDefaultX2());
	return cupnp::mutatePointerField<cupnp::List<double>, 1>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoil::PLFilament::nonDefaultX2() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<double> fsc::devices::w7x::cu::CoilsDBCoil::PLFilament::getX2() const { 
	return cupnp::getPointerField<cupnp::List<double>, 1>(structure, data, reinterpret_cast<const capnp::word*>(X2_DEFAULT_VALUE));
} 

cupnp::List<double> fsc::devices::w7x::cu::CoilsDBCoil::PLFilament::mutateX3() { 
	CUPNP_REQUIRE(nonDefaultX3());
	return cupnp::mutatePointerField<cupnp::List<double>, 2>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoil::PLFilament::nonDefaultX3() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::List<double> fsc::devices::w7x::cu::CoilsDBCoil::PLFilament::getX3() const { 
	return cupnp::getPointerField<cupnp::List<double>, 2>(structure, data, reinterpret_cast<const capnp::word*>(X3_DEFAULT_VALUE));
} 

// ===== struct fsc::devices::w7x::cu::CoilsDBCoil =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::CoilsDBCoil> { using Type = fsc::devices::w7x::cu::CoilsDBCoil; }; 
} // namespace ::cupnp

fsc::devices::w7x::cu::CoilsDBCoil::PLFilament fsc::devices::w7x::cu::CoilsDBCoil::mutatePolylineFilament() { 
	CUPNP_REQUIRE(nonDefaultPolylineFilament());
	return cupnp::mutatePointerField<fsc::devices::w7x::cu::CoilsDBCoil::PLFilament, 0>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoil::nonDefaultPolylineFilament() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::devices::w7x::cu::CoilsDBCoil::PLFilament fsc::devices::w7x::cu::CoilsDBCoil::getPolylineFilament() const { 
	return cupnp::getPointerField<fsc::devices::w7x::cu::CoilsDBCoil::PLFilament, 0>(structure, data, reinterpret_cast<const capnp::word*>(POLYLINE_FILAMENT_DEFAULT_VALUE));
} 

// ===== struct fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::CoilsDBCoilInfo::HistoryEntry> { using Type = fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry; }; 
} // namespace ::cupnp

void fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::setTimeStamp(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 0>(structure, data, 0, newVal);
} 

uint64_t fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::getTimeStamp() const { 
	return cupnp::getPrimitiveField<uint64_t, 0>(structure, data, 0);
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::mutateAuthor() { 
	CUPNP_REQUIRE(nonDefaultAuthor());
	return cupnp::mutatePointerField<cupnp::Text, 0>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::nonDefaultAuthor() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::getAuthor() const { 
	return cupnp::getPointerField<cupnp::Text, 0>(structure, data, reinterpret_cast<const capnp::word*>(AUTHOR_DEFAULT_VALUE));
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::mutateMethod() { 
	CUPNP_REQUIRE(nonDefaultMethod());
	return cupnp::mutatePointerField<cupnp::Text, 1>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::nonDefaultMethod() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::getMethod() const { 
	return cupnp::getPointerField<cupnp::Text, 1>(structure, data, reinterpret_cast<const capnp::word*>(METHOD_DEFAULT_VALUE));
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::mutateComment() { 
	CUPNP_REQUIRE(nonDefaultComment());
	return cupnp::mutatePointerField<cupnp::Text, 2>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::nonDefaultComment() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::HistoryEntry::getComment() const { 
	return cupnp::getPointerField<cupnp::Text, 2>(structure, data, reinterpret_cast<const capnp::word*>(COMMENT_DEFAULT_VALUE));
} 

// ===== struct fsc::devices::w7x::cu::CoilsDBCoilInfo =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::CoilsDBCoilInfo> { using Type = fsc::devices::w7x::cu::CoilsDBCoilInfo; }; 
} // namespace ::cupnp

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::mutateName() { 
	CUPNP_REQUIRE(nonDefaultName());
	return cupnp::mutatePointerField<cupnp::Text, 0>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::nonDefaultName() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::getName() const { 
	return cupnp::getPointerField<cupnp::Text, 0>(structure, data, reinterpret_cast<const capnp::word*>(NAME_DEFAULT_VALUE));
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::mutateMachine() { 
	CUPNP_REQUIRE(nonDefaultMachine());
	return cupnp::mutatePointerField<cupnp::Text, 1>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::nonDefaultMachine() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::getMachine() const { 
	return cupnp::getPointerField<cupnp::Text, 1>(structure, data, reinterpret_cast<const capnp::word*>(MACHINE_DEFAULT_VALUE));
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::mutateState() { 
	CUPNP_REQUIRE(nonDefaultState());
	return cupnp::mutatePointerField<cupnp::Text, 2>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::nonDefaultState() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::getState() const { 
	return cupnp::getPointerField<cupnp::Text, 2>(structure, data, reinterpret_cast<const capnp::word*>(STATE_DEFAULT_VALUE));
} 

void fsc::devices::w7x::cu::CoilsDBCoilInfo::setQuality(int64_t newVal) { 
	cupnp::setPrimitiveField<int64_t, 0>(structure, data, 0, newVal);
} 

int64_t fsc::devices::w7x::cu::CoilsDBCoilInfo::getQuality() const { 
	return cupnp::getPrimitiveField<int64_t, 0>(structure, data, 0);
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::mutateAuthor() { 
	CUPNP_REQUIRE(nonDefaultAuthor());
	return cupnp::mutatePointerField<cupnp::Text, 3>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::nonDefaultAuthor() const { 
	return cupnp::hasPointerField<3>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::getAuthor() const { 
	return cupnp::getPointerField<cupnp::Text, 3>(structure, data, reinterpret_cast<const capnp::word*>(AUTHOR_DEFAULT_VALUE));
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::mutateLocation() { 
	CUPNP_REQUIRE(nonDefaultLocation());
	return cupnp::mutatePointerField<cupnp::Text, 4>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::nonDefaultLocation() const { 
	return cupnp::hasPointerField<4>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::getLocation() const { 
	return cupnp::getPointerField<cupnp::Text, 4>(structure, data, reinterpret_cast<const capnp::word*>(LOCATION_DEFAULT_VALUE));
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::mutateId() { 
	CUPNP_REQUIRE(nonDefaultId());
	return cupnp::mutatePointerField<cupnp::Text, 5>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::nonDefaultId() const { 
	return cupnp::hasPointerField<5>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::getId() const { 
	return cupnp::getPointerField<cupnp::Text, 5>(structure, data, reinterpret_cast<const capnp::word*>(ID_DEFAULT_VALUE));
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::mutateMethod() { 
	CUPNP_REQUIRE(nonDefaultMethod());
	return cupnp::mutatePointerField<cupnp::Text, 6>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::nonDefaultMethod() const { 
	return cupnp::hasPointerField<6>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::getMethod() const { 
	return cupnp::getPointerField<cupnp::Text, 6>(structure, data, reinterpret_cast<const capnp::word*>(METHOD_DEFAULT_VALUE));
} 

cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::mutateComment() { 
	CUPNP_REQUIRE(nonDefaultComment());
	return cupnp::mutatePointerField<cupnp::Text, 7>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBCoilInfo::nonDefaultComment() const { 
	return cupnp::hasPointerField<7>(structure, data);
} 

const cupnp::Text fsc::devices::w7x::cu::CoilsDBCoilInfo::getComment() const { 
	return cupnp::getPointerField<cupnp::Text, 7>(structure, data, reinterpret_cast<const capnp::word*>(COMMENT_DEFAULT_VALUE));
} 

// ===== struct fsc::devices::w7x::cu::CoilsDBConfig =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::devices::w7x::CoilsDBConfig> { using Type = fsc::devices::w7x::cu::CoilsDBConfig; }; 
} // namespace ::cupnp

cupnp::List<uint64_t> fsc::devices::w7x::cu::CoilsDBConfig::mutateCoils() { 
	CUPNP_REQUIRE(nonDefaultCoils());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBConfig::nonDefaultCoils() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<uint64_t> fsc::devices::w7x::cu::CoilsDBConfig::getCoils() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(COILS_DEFAULT_VALUE));
} 

cupnp::List<double> fsc::devices::w7x::cu::CoilsDBConfig::mutateCurrents() { 
	CUPNP_REQUIRE(nonDefaultCurrents());
	return cupnp::mutatePointerField<cupnp::List<double>, 1>(structure, data);
} 

bool fsc::devices::w7x::cu::CoilsDBConfig::nonDefaultCurrents() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<double> fsc::devices::w7x::cu::CoilsDBConfig::getCurrents() const { 
	return cupnp::getPointerField<cupnp::List<double>, 1>(structure, data, reinterpret_cast<const capnp::word*>(CURRENTS_DEFAULT_VALUE));
} 

void fsc::devices::w7x::cu::CoilsDBConfig::setScale(double newVal) { 
	cupnp::setPrimitiveField<double, 0>(structure, data, 0, newVal);
} 

double fsc::devices::w7x::cu::CoilsDBConfig::getScale() const { 
	return cupnp::getPrimitiveField<double, 0>(structure, data, 0);
} 

