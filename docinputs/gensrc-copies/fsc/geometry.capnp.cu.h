#pragma once 

#include <cupnp/cupnp.h>
#include "geometry.capnp.h"
#include "data.capnp.cu.h"

namespace fsc{
namespace cu{

struct GeometryResolver{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	GeometryResolver(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct GeometryLib{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	GeometryLib(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct TagValue{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline TagValue(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline bool hasNotSet() const;
	inline uint64_t getUInt64() const;
	inline void setUInt64(uint64_t newVal);

	inline bool hasUInt64() const;
}; // struct fsc::cu::TagValue

struct Tag{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Tag(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char NAME_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateName();

	inline const cupnp::Text getName() const;
	inline bool nonDefaultName() const;

	inline static const unsigned char VALUE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::TagValue mutateValue();

	inline const fsc::cu::TagValue getValue() const;
	inline bool nonDefaultValue() const;

}; // struct fsc::cu::Tag

struct CartesianGrid{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline CartesianGrid(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline double getXMin() const;
	inline void setXMin(double newVal);

	inline double getXMax() const;
	inline void setXMax(double newVal);

	inline double getYMin() const;
	inline void setYMin(double newVal);

	inline double getYMax() const;
	inline void setYMax(double newVal);

	inline double getZMin() const;
	inline void setZMin(double newVal);

	inline double getZMax() const;
	inline void setZMax(double newVal);

	inline uint32_t getNX() const;
	inline void setNX(uint32_t newVal);

	inline uint32_t getNY() const;
	inline void setNY(uint32_t newVal);

	inline uint32_t getNZ() const;
	inline void setNZ(uint32_t newVal);

}; // struct fsc::cu::CartesianGrid

template<typename Param0T>
struct Transformed{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Transformed(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct Shifted{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Shifted(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char SHIFT_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateShift();

	inline const cupnp::List<double> getShift() const;
	inline bool nonDefaultShift() const;

	inline static const unsigned char NODE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Transformed<Param0T> mutateNode();

	inline const fsc::cu::Transformed<Param0T> getNode() const;
	inline bool nonDefaultNode() const;

}; // struct typename fsc::cu::Transformed<Param0T>::Shifted

struct Turned{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Turned(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline double getAngle() const;
	inline void setAngle(double newVal);

	inline static const unsigned char CENTER_DEFAULT_VALUE [] = {1, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateCenter();

	inline const cupnp::List<double> getCenter() const;
	inline bool nonDefaultCenter() const;

	inline static const unsigned char AXIS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateAxis();

	inline const cupnp::List<double> getAxis() const;
	inline bool nonDefaultAxis() const;

	inline static const unsigned char NODE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Transformed<Param0T> mutateNode();

	inline const fsc::cu::Transformed<Param0T> getNode() const;
	inline bool nonDefaultNode() const;

}; // struct typename fsc::cu::Transformed<Param0T>::Turned

	inline static const unsigned char LEAF_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline Param0T mutateLeaf();

	inline const Param0T getLeaf() const;
	inline bool nonDefaultLeaf() const;
	inline bool hasLeaf() const;

	inline bool isShifted() const;
	
	inline typename fsc::cu::Transformed<Param0T>::Shifted getShifted() const;

	inline bool isTurned() const;
	
	inline typename fsc::cu::Transformed<Param0T>::Turned getTurned() const;

}; // struct fsc::cu::Transformed<Param0T>

struct Mesh{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Mesh(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char VERTICES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Float64Tensor mutateVertices();

	inline const fsc::cu::Float64Tensor getVertices() const;
	inline bool nonDefaultVertices() const;

	inline static const unsigned char INDICES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint32_t> mutateIndices();

	inline const cupnp::List<uint32_t> getIndices() const;
	inline bool nonDefaultIndices() const;

	inline static const unsigned char POLY_MESH_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint32_t> mutatePolyMesh();

	inline const cupnp::List<uint32_t> getPolyMesh() const;
	inline bool nonDefaultPolyMesh() const;
	inline bool hasPolyMesh() const;

	inline bool hasTriMesh() const;
}; // struct fsc::cu::Mesh

struct Geometry{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Geometry(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char TAGS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::Tag> mutateTags();

	inline const cupnp::List<fsc::cu::Tag> getTags() const;
	inline bool nonDefaultTags() const;

	inline static const unsigned char COMBINED_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::Geometry> mutateCombined();

	inline const cupnp::List<fsc::cu::Geometry> getCombined() const;
	inline bool nonDefaultCombined() const;
	inline bool hasCombined() const;

	inline static const unsigned char TRANSFORMED_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Transformed<fsc::cu::Geometry> mutateTransformed();

	inline const fsc::cu::Transformed<fsc::cu::Geometry> getTransformed() const;
	inline bool nonDefaultTransformed() const;
	inline bool hasTransformed() const;

	inline static const unsigned char REF_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::DataRef<fsc::cu::Geometry> mutateRef();

	inline const fsc::cu::DataRef<fsc::cu::Geometry> getRef() const;
	inline bool nonDefaultRef() const;
	inline bool hasRef() const;

	inline static const unsigned char MESH_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::DataRef<fsc::cu::Mesh> mutateMesh();

	inline const fsc::cu::DataRef<fsc::cu::Mesh> getMesh() const;
	inline bool nonDefaultMesh() const;
	inline bool hasMesh() const;

	inline static const unsigned char COMPONENTS_D_B_MESHES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateComponentsDBMeshes();

	inline const cupnp::List<uint64_t> getComponentsDBMeshes() const;
	inline bool nonDefaultComponentsDBMeshes() const;
	inline bool hasComponentsDBMeshes() const;

	inline static const unsigned char COMPONENTS_D_B_ASSEMBLIES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateComponentsDBAssemblies();

	inline const cupnp::List<uint64_t> getComponentsDBAssemblies() const;
	inline bool nonDefaultComponentsDBAssemblies() const;
	inline bool hasComponentsDBAssemblies() const;

}; // struct fsc::cu::Geometry

struct MergedGeometry{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline MergedGeometry(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct Entry{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Entry(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char TAGS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::TagValue> mutateTags();

	inline const cupnp::List<fsc::cu::TagValue> getTags() const;
	inline bool nonDefaultTags() const;

	inline static const unsigned char MESH_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Mesh mutateMesh();

	inline const fsc::cu::Mesh getMesh() const;
	inline bool nonDefaultMesh() const;

}; // struct fsc::cu::MergedGeometry::Entry

	inline static const unsigned char TAG_NAMES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<cupnp::Text> mutateTagNames();

	inline const cupnp::List<cupnp::Text> getTagNames() const;
	inline bool nonDefaultTagNames() const;

	inline static const unsigned char ENTRIES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::MergedGeometry::Entry> mutateEntries();

	inline const cupnp::List<fsc::cu::MergedGeometry::Entry> getEntries() const;
	inline bool nonDefaultEntries() const;

}; // struct fsc::cu::MergedGeometry

struct IndexedGeometry{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline IndexedGeometry(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct ElementRef{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline ElementRef(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline uint64_t getMeshIndex() const;
	inline void setMeshIndex(uint64_t newVal);

	inline uint64_t getElementIndex() const;
	inline void setElementIndex(uint64_t newVal);

}; // struct fsc::cu::IndexedGeometry::ElementRef

struct GridEntry{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline GridEntry(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char ELEMENTS_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::IndexedGeometry::ElementRef> mutateElements();

	inline const cupnp::List<fsc::cu::IndexedGeometry::ElementRef> getElements() const;
	inline bool nonDefaultElements() const;

}; // struct fsc::cu::IndexedGeometry::GridEntry

	inline static const unsigned char BASE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::DataRef<fsc::cu::MergedGeometry> mutateBase();

	inline const fsc::cu::DataRef<fsc::cu::MergedGeometry> getBase() const;
	inline bool nonDefaultBase() const;

	inline static const unsigned char GRID_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::CartesianGrid mutateGrid();

	inline const fsc::cu::CartesianGrid getGrid() const;
	inline bool nonDefaultGrid() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::ShapedList<cupnp::List<cupnp::List<fsc::cu::IndexedGeometry::ElementRef>>> mutateData();

	inline const fsc::cu::ShapedList<cupnp::List<cupnp::List<fsc::cu::IndexedGeometry::ElementRef>>> getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::IndexedGeometry


}}
// ===== struct fsc::cu::TagValue =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::TagValue> { using Type = fsc::cu::TagValue; }; 
} // namespace ::cupnp

bool fsc::cu::TagValue::hasNotSet() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0;
} 

void fsc::cu::TagValue::setUInt64(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 1>(structure, data, 0, newVal);
	cupnp::setDiscriminant<0>(structure, data, 1);
} 

uint64_t fsc::cu::TagValue::getUInt64() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 1)
		return 0;
	
	return cupnp::getPrimitiveField<uint64_t, 1>(structure, data, 0);
} 

bool fsc::cu::TagValue::hasUInt64() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1;
} 

// ===== struct fsc::cu::Tag =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Tag> { using Type = fsc::cu::Tag; }; 
} // namespace ::cupnp

cupnp::Text fsc::cu::Tag::mutateName() { 
	CUPNP_REQUIRE(nonDefaultName());
	return cupnp::mutatePointerField<cupnp::Text, 0>(structure, data);
} 

bool fsc::cu::Tag::nonDefaultName() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::Text fsc::cu::Tag::getName() const { 
	return cupnp::getPointerField<cupnp::Text, 0>(structure, data, reinterpret_cast<const capnp::word*>(NAME_DEFAULT_VALUE));
} 

fsc::cu::TagValue fsc::cu::Tag::mutateValue() { 
	CUPNP_REQUIRE(nonDefaultValue());
	return cupnp::mutatePointerField<fsc::cu::TagValue, 1>(structure, data);
} 

bool fsc::cu::Tag::nonDefaultValue() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const fsc::cu::TagValue fsc::cu::Tag::getValue() const { 
	return cupnp::getPointerField<fsc::cu::TagValue, 1>(structure, data, reinterpret_cast<const capnp::word*>(VALUE_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::CartesianGrid =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::CartesianGrid> { using Type = fsc::cu::CartesianGrid; }; 
} // namespace ::cupnp

void fsc::cu::CartesianGrid::setXMin(double newVal) { 
	cupnp::setPrimitiveField<double, 0>(structure, data, 0, newVal);
} 

double fsc::cu::CartesianGrid::getXMin() const { 
	return cupnp::getPrimitiveField<double, 0>(structure, data, 0);
} 

void fsc::cu::CartesianGrid::setXMax(double newVal) { 
	cupnp::setPrimitiveField<double, 1>(structure, data, 0, newVal);
} 

double fsc::cu::CartesianGrid::getXMax() const { 
	return cupnp::getPrimitiveField<double, 1>(structure, data, 0);
} 

void fsc::cu::CartesianGrid::setYMin(double newVal) { 
	cupnp::setPrimitiveField<double, 2>(structure, data, 0, newVal);
} 

double fsc::cu::CartesianGrid::getYMin() const { 
	return cupnp::getPrimitiveField<double, 2>(structure, data, 0);
} 

void fsc::cu::CartesianGrid::setYMax(double newVal) { 
	cupnp::setPrimitiveField<double, 3>(structure, data, 0, newVal);
} 

double fsc::cu::CartesianGrid::getYMax() const { 
	return cupnp::getPrimitiveField<double, 3>(structure, data, 0);
} 

void fsc::cu::CartesianGrid::setZMin(double newVal) { 
	cupnp::setPrimitiveField<double, 4>(structure, data, 0, newVal);
} 

double fsc::cu::CartesianGrid::getZMin() const { 
	return cupnp::getPrimitiveField<double, 4>(structure, data, 0);
} 

void fsc::cu::CartesianGrid::setZMax(double newVal) { 
	cupnp::setPrimitiveField<double, 5>(structure, data, 0, newVal);
} 

double fsc::cu::CartesianGrid::getZMax() const { 
	return cupnp::getPrimitiveField<double, 5>(structure, data, 0);
} 

void fsc::cu::CartesianGrid::setNX(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 12>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::CartesianGrid::getNX() const { 
	return cupnp::getPrimitiveField<uint32_t, 12>(structure, data, 0);
} 

void fsc::cu::CartesianGrid::setNY(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 13>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::CartesianGrid::getNY() const { 
	return cupnp::getPrimitiveField<uint32_t, 13>(structure, data, 0);
} 

void fsc::cu::CartesianGrid::setNZ(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 14>(structure, data, 0, newVal);
} 

uint32_t fsc::cu::CartesianGrid::getNZ() const { 
	return cupnp::getPrimitiveField<uint32_t, 14>(structure, data, 0);
} 

// ===== struct typename fsc::cu::Transformed<Param0T>::Shifted =====

template<typename Param0T>
cupnp::List<double> fsc::cu::Transformed<Param0T>::Shifted::mutateShift() { 
	CUPNP_REQUIRE(nonDefaultShift());
	return cupnp::mutatePointerField<cupnp::List<double>, 0>(structure, data);
} 

template<typename Param0T>
bool fsc::cu::Transformed<Param0T>::Shifted::nonDefaultShift() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

template<typename Param0T>
const cupnp::List<double> fsc::cu::Transformed<Param0T>::Shifted::getShift() const { 
	return cupnp::getPointerField<cupnp::List<double>, 0>(structure, data, reinterpret_cast<const capnp::word*>(SHIFT_DEFAULT_VALUE));
} 

template<typename Param0T>
fsc::cu::Transformed<Param0T> fsc::cu::Transformed<Param0T>::Shifted::mutateNode() { 
	CUPNP_REQUIRE(nonDefaultNode());
	return cupnp::mutatePointerField<fsc::cu::Transformed<Param0T>, 1>(structure, data);
} 

template<typename Param0T>
bool fsc::cu::Transformed<Param0T>::Shifted::nonDefaultNode() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

template<typename Param0T>
const fsc::cu::Transformed<Param0T> fsc::cu::Transformed<Param0T>::Shifted::getNode() const { 
	return cupnp::getPointerField<fsc::cu::Transformed<Param0T>, 1>(structure, data, reinterpret_cast<const capnp::word*>(NODE_DEFAULT_VALUE));
} 

// ===== struct typename fsc::cu::Transformed<Param0T>::Turned =====

template<typename Param0T>
void fsc::cu::Transformed<Param0T>::Turned::setAngle(double newVal) { 
	cupnp::setPrimitiveField<double, 1>(structure, data, 0, newVal);
} 

template<typename Param0T>
double fsc::cu::Transformed<Param0T>::Turned::getAngle() const { 
	return cupnp::getPrimitiveField<double, 1>(structure, data, 0);
} 

template<typename Param0T>
cupnp::List<double> fsc::cu::Transformed<Param0T>::Turned::mutateCenter() { 
	CUPNP_REQUIRE(nonDefaultCenter());
	return cupnp::mutatePointerField<cupnp::List<double>, 0>(structure, data);
} 

template<typename Param0T>
bool fsc::cu::Transformed<Param0T>::Turned::nonDefaultCenter() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

template<typename Param0T>
const cupnp::List<double> fsc::cu::Transformed<Param0T>::Turned::getCenter() const { 
	return cupnp::getPointerField<cupnp::List<double>, 0>(structure, data, reinterpret_cast<const capnp::word*>(CENTER_DEFAULT_VALUE));
} 

template<typename Param0T>
cupnp::List<double> fsc::cu::Transformed<Param0T>::Turned::mutateAxis() { 
	CUPNP_REQUIRE(nonDefaultAxis());
	return cupnp::mutatePointerField<cupnp::List<double>, 1>(structure, data);
} 

template<typename Param0T>
bool fsc::cu::Transformed<Param0T>::Turned::nonDefaultAxis() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

template<typename Param0T>
const cupnp::List<double> fsc::cu::Transformed<Param0T>::Turned::getAxis() const { 
	return cupnp::getPointerField<cupnp::List<double>, 1>(structure, data, reinterpret_cast<const capnp::word*>(AXIS_DEFAULT_VALUE));
} 

template<typename Param0T>
fsc::cu::Transformed<Param0T> fsc::cu::Transformed<Param0T>::Turned::mutateNode() { 
	CUPNP_REQUIRE(nonDefaultNode());
	return cupnp::mutatePointerField<fsc::cu::Transformed<Param0T>, 2>(structure, data);
} 

template<typename Param0T>
bool fsc::cu::Transformed<Param0T>::Turned::nonDefaultNode() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

template<typename Param0T>
const fsc::cu::Transformed<Param0T> fsc::cu::Transformed<Param0T>::Turned::getNode() const { 
	return cupnp::getPointerField<fsc::cu::Transformed<Param0T>, 2>(structure, data, reinterpret_cast<const capnp::word*>(NODE_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::Transformed<Param0T> =====

// CuFor specializaation
namespace cupnp {
template<typename Param0T>
struct CuFor_<fsc::Transformed<Param0T>> { using Type = fsc::cu::Transformed<Param0T>; }; 
} // namespace ::cupnp

template<typename Param0T>
Param0T fsc::cu::Transformed<Param0T>::mutateLeaf() { 
	CUPNP_REQUIRE(nonDefaultLeaf());
	return cupnp::mutatePointerField<Param0T, 0>(structure, data);
} 

template<typename Param0T>
bool fsc::cu::Transformed<Param0T>::hasLeaf() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0;
} 

template<typename Param0T>
bool fsc::cu::Transformed<Param0T>::nonDefaultLeaf() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0 && cupnp::hasPointerField<0>(structure, data);
} 

template<typename Param0T>
const Param0T fsc::cu::Transformed<Param0T>::getLeaf() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 0)
		return cupnp::getPointer<Param0T>(reinterpret_cast<const capnp::word*>(LEAF_DEFAULT_VALUE));
	
	return cupnp::getPointerField<Param0T, 0>(structure, data, reinterpret_cast<const capnp::word*>(LEAF_DEFAULT_VALUE));
} 

template<typename Param0T>
bool fsc::cu::Transformed<Param0T>::isShifted() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1;
} 

template<typename Param0T>
typename fsc::cu::Transformed<Param0T>::Shifted fsc::cu::Transformed<Param0T>::getShifted() const { 
	return typename fsc::cu::Transformed<Param0T>::Shifted(structure, data);
} 

template<typename Param0T>
bool fsc::cu::Transformed<Param0T>::isTurned() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 2;
} 

template<typename Param0T>
typename fsc::cu::Transformed<Param0T>::Turned fsc::cu::Transformed<Param0T>::getTurned() const { 
	return typename fsc::cu::Transformed<Param0T>::Turned(structure, data);
} 

// ===== struct fsc::cu::Mesh =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Mesh> { using Type = fsc::cu::Mesh; }; 
} // namespace ::cupnp

fsc::cu::Float64Tensor fsc::cu::Mesh::mutateVertices() { 
	CUPNP_REQUIRE(nonDefaultVertices());
	return cupnp::mutatePointerField<fsc::cu::Float64Tensor, 0>(structure, data);
} 

bool fsc::cu::Mesh::nonDefaultVertices() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::Float64Tensor fsc::cu::Mesh::getVertices() const { 
	return cupnp::getPointerField<fsc::cu::Float64Tensor, 0>(structure, data, reinterpret_cast<const capnp::word*>(VERTICES_DEFAULT_VALUE));
} 

cupnp::List<uint32_t> fsc::cu::Mesh::mutateIndices() { 
	CUPNP_REQUIRE(nonDefaultIndices());
	return cupnp::mutatePointerField<cupnp::List<uint32_t>, 1>(structure, data);
} 

bool fsc::cu::Mesh::nonDefaultIndices() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<uint32_t> fsc::cu::Mesh::getIndices() const { 
	return cupnp::getPointerField<cupnp::List<uint32_t>, 1>(structure, data, reinterpret_cast<const capnp::word*>(INDICES_DEFAULT_VALUE));
} 

cupnp::List<uint32_t> fsc::cu::Mesh::mutatePolyMesh() { 
	CUPNP_REQUIRE(nonDefaultPolyMesh());
	return cupnp::mutatePointerField<cupnp::List<uint32_t>, 2>(structure, data);
} 

bool fsc::cu::Mesh::hasPolyMesh() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0;
} 

bool fsc::cu::Mesh::nonDefaultPolyMesh() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0 && cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::List<uint32_t> fsc::cu::Mesh::getPolyMesh() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 0)
		return cupnp::getPointer<cupnp::List<uint32_t>>(reinterpret_cast<const capnp::word*>(POLY_MESH_DEFAULT_VALUE));
	
	return cupnp::getPointerField<cupnp::List<uint32_t>, 2>(structure, data, reinterpret_cast<const capnp::word*>(POLY_MESH_DEFAULT_VALUE));
} 

bool fsc::cu::Mesh::hasTriMesh() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1;
} 

// ===== struct fsc::cu::Geometry =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Geometry> { using Type = fsc::cu::Geometry; }; 
} // namespace ::cupnp

cupnp::List<fsc::cu::Tag> fsc::cu::Geometry::mutateTags() { 
	CUPNP_REQUIRE(nonDefaultTags());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::Tag>, 0>(structure, data);
} 

bool fsc::cu::Geometry::nonDefaultTags() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<fsc::cu::Tag> fsc::cu::Geometry::getTags() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::Tag>, 0>(structure, data, reinterpret_cast<const capnp::word*>(TAGS_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::Geometry> fsc::cu::Geometry::mutateCombined() { 
	CUPNP_REQUIRE(nonDefaultCombined());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::Geometry>, 1>(structure, data);
} 

bool fsc::cu::Geometry::hasCombined() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0;
} 

bool fsc::cu::Geometry::nonDefaultCombined() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0 && cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<fsc::cu::Geometry> fsc::cu::Geometry::getCombined() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 0)
		return cupnp::getPointer<cupnp::List<fsc::cu::Geometry>>(reinterpret_cast<const capnp::word*>(COMBINED_DEFAULT_VALUE));
	
	return cupnp::getPointerField<cupnp::List<fsc::cu::Geometry>, 1>(structure, data, reinterpret_cast<const capnp::word*>(COMBINED_DEFAULT_VALUE));
} 

fsc::cu::Transformed<fsc::cu::Geometry> fsc::cu::Geometry::mutateTransformed() { 
	CUPNP_REQUIRE(nonDefaultTransformed());
	return cupnp::mutatePointerField<fsc::cu::Transformed<fsc::cu::Geometry>, 1>(structure, data);
} 

bool fsc::cu::Geometry::hasTransformed() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1;
} 

bool fsc::cu::Geometry::nonDefaultTransformed() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1 && cupnp::hasPointerField<1>(structure, data);
} 

const fsc::cu::Transformed<fsc::cu::Geometry> fsc::cu::Geometry::getTransformed() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 1)
		return cupnp::getPointer<fsc::cu::Transformed<fsc::cu::Geometry>>(reinterpret_cast<const capnp::word*>(TRANSFORMED_DEFAULT_VALUE));
	
	return cupnp::getPointerField<fsc::cu::Transformed<fsc::cu::Geometry>, 1>(structure, data, reinterpret_cast<const capnp::word*>(TRANSFORMED_DEFAULT_VALUE));
} 

fsc::cu::DataRef<fsc::cu::Geometry> fsc::cu::Geometry::mutateRef() { 
	CUPNP_REQUIRE(nonDefaultRef());
	return cupnp::mutatePointerField<fsc::cu::DataRef<fsc::cu::Geometry>, 1>(structure, data);
} 

bool fsc::cu::Geometry::hasRef() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 2;
} 

bool fsc::cu::Geometry::nonDefaultRef() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 2 && cupnp::hasPointerField<1>(structure, data);
} 

const fsc::cu::DataRef<fsc::cu::Geometry> fsc::cu::Geometry::getRef() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 2)
		return cupnp::getPointer<fsc::cu::DataRef<fsc::cu::Geometry>>(reinterpret_cast<const capnp::word*>(REF_DEFAULT_VALUE));
	
	return cupnp::getPointerField<fsc::cu::DataRef<fsc::cu::Geometry>, 1>(structure, data, reinterpret_cast<const capnp::word*>(REF_DEFAULT_VALUE));
} 

fsc::cu::DataRef<fsc::cu::Mesh> fsc::cu::Geometry::mutateMesh() { 
	CUPNP_REQUIRE(nonDefaultMesh());
	return cupnp::mutatePointerField<fsc::cu::DataRef<fsc::cu::Mesh>, 1>(structure, data);
} 

bool fsc::cu::Geometry::hasMesh() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 3;
} 

bool fsc::cu::Geometry::nonDefaultMesh() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 3 && cupnp::hasPointerField<1>(structure, data);
} 

const fsc::cu::DataRef<fsc::cu::Mesh> fsc::cu::Geometry::getMesh() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 3)
		return cupnp::getPointer<fsc::cu::DataRef<fsc::cu::Mesh>>(reinterpret_cast<const capnp::word*>(MESH_DEFAULT_VALUE));
	
	return cupnp::getPointerField<fsc::cu::DataRef<fsc::cu::Mesh>, 1>(structure, data, reinterpret_cast<const capnp::word*>(MESH_DEFAULT_VALUE));
} 

cupnp::List<uint64_t> fsc::cu::Geometry::mutateComponentsDBMeshes() { 
	CUPNP_REQUIRE(nonDefaultComponentsDBMeshes());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 1>(structure, data);
} 

bool fsc::cu::Geometry::hasComponentsDBMeshes() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 4;
} 

bool fsc::cu::Geometry::nonDefaultComponentsDBMeshes() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 4 && cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::Geometry::getComponentsDBMeshes() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 4)
		return cupnp::getPointer<cupnp::List<uint64_t>>(reinterpret_cast<const capnp::word*>(COMPONENTS_D_B_MESHES_DEFAULT_VALUE));
	
	return cupnp::getPointerField<cupnp::List<uint64_t>, 1>(structure, data, reinterpret_cast<const capnp::word*>(COMPONENTS_D_B_MESHES_DEFAULT_VALUE));
} 

cupnp::List<uint64_t> fsc::cu::Geometry::mutateComponentsDBAssemblies() { 
	CUPNP_REQUIRE(nonDefaultComponentsDBAssemblies());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 1>(structure, data);
} 

bool fsc::cu::Geometry::hasComponentsDBAssemblies() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 5;
} 

bool fsc::cu::Geometry::nonDefaultComponentsDBAssemblies() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 5 && cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::Geometry::getComponentsDBAssemblies() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 5)
		return cupnp::getPointer<cupnp::List<uint64_t>>(reinterpret_cast<const capnp::word*>(COMPONENTS_D_B_ASSEMBLIES_DEFAULT_VALUE));
	
	return cupnp::getPointerField<cupnp::List<uint64_t>, 1>(structure, data, reinterpret_cast<const capnp::word*>(COMPONENTS_D_B_ASSEMBLIES_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::MergedGeometry::Entry =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::MergedGeometry::Entry> { using Type = fsc::cu::MergedGeometry::Entry; }; 
} // namespace ::cupnp

cupnp::List<fsc::cu::TagValue> fsc::cu::MergedGeometry::Entry::mutateTags() { 
	CUPNP_REQUIRE(nonDefaultTags());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::TagValue>, 0>(structure, data);
} 

bool fsc::cu::MergedGeometry::Entry::nonDefaultTags() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<fsc::cu::TagValue> fsc::cu::MergedGeometry::Entry::getTags() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::TagValue>, 0>(structure, data, reinterpret_cast<const capnp::word*>(TAGS_DEFAULT_VALUE));
} 

fsc::cu::Mesh fsc::cu::MergedGeometry::Entry::mutateMesh() { 
	CUPNP_REQUIRE(nonDefaultMesh());
	return cupnp::mutatePointerField<fsc::cu::Mesh, 1>(structure, data);
} 

bool fsc::cu::MergedGeometry::Entry::nonDefaultMesh() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const fsc::cu::Mesh fsc::cu::MergedGeometry::Entry::getMesh() const { 
	return cupnp::getPointerField<fsc::cu::Mesh, 1>(structure, data, reinterpret_cast<const capnp::word*>(MESH_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::MergedGeometry =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::MergedGeometry> { using Type = fsc::cu::MergedGeometry; }; 
} // namespace ::cupnp

cupnp::List<cupnp::Text> fsc::cu::MergedGeometry::mutateTagNames() { 
	CUPNP_REQUIRE(nonDefaultTagNames());
	return cupnp::mutatePointerField<cupnp::List<cupnp::Text>, 0>(structure, data);
} 

bool fsc::cu::MergedGeometry::nonDefaultTagNames() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<cupnp::Text> fsc::cu::MergedGeometry::getTagNames() const { 
	return cupnp::getPointerField<cupnp::List<cupnp::Text>, 0>(structure, data, reinterpret_cast<const capnp::word*>(TAG_NAMES_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::MergedGeometry::Entry> fsc::cu::MergedGeometry::mutateEntries() { 
	CUPNP_REQUIRE(nonDefaultEntries());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::MergedGeometry::Entry>, 1>(structure, data);
} 

bool fsc::cu::MergedGeometry::nonDefaultEntries() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<fsc::cu::MergedGeometry::Entry> fsc::cu::MergedGeometry::getEntries() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::MergedGeometry::Entry>, 1>(structure, data, reinterpret_cast<const capnp::word*>(ENTRIES_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::IndexedGeometry::ElementRef =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::IndexedGeometry::ElementRef> { using Type = fsc::cu::IndexedGeometry::ElementRef; }; 
} // namespace ::cupnp

void fsc::cu::IndexedGeometry::ElementRef::setMeshIndex(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 0>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::IndexedGeometry::ElementRef::getMeshIndex() const { 
	return cupnp::getPrimitiveField<uint64_t, 0>(structure, data, 0);
} 

void fsc::cu::IndexedGeometry::ElementRef::setElementIndex(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 1>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::IndexedGeometry::ElementRef::getElementIndex() const { 
	return cupnp::getPrimitiveField<uint64_t, 1>(structure, data, 0);
} 

// ===== struct fsc::cu::IndexedGeometry::GridEntry =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::IndexedGeometry::GridEntry> { using Type = fsc::cu::IndexedGeometry::GridEntry; }; 
} // namespace ::cupnp

cupnp::List<fsc::cu::IndexedGeometry::ElementRef> fsc::cu::IndexedGeometry::GridEntry::mutateElements() { 
	CUPNP_REQUIRE(nonDefaultElements());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::IndexedGeometry::ElementRef>, 0>(structure, data);
} 

bool fsc::cu::IndexedGeometry::GridEntry::nonDefaultElements() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<fsc::cu::IndexedGeometry::ElementRef> fsc::cu::IndexedGeometry::GridEntry::getElements() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::IndexedGeometry::ElementRef>, 0>(structure, data, reinterpret_cast<const capnp::word*>(ELEMENTS_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::IndexedGeometry =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::IndexedGeometry> { using Type = fsc::cu::IndexedGeometry; }; 
} // namespace ::cupnp

fsc::cu::DataRef<fsc::cu::MergedGeometry> fsc::cu::IndexedGeometry::mutateBase() { 
	CUPNP_REQUIRE(nonDefaultBase());
	return cupnp::mutatePointerField<fsc::cu::DataRef<fsc::cu::MergedGeometry>, 0>(structure, data);
} 

bool fsc::cu::IndexedGeometry::nonDefaultBase() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::DataRef<fsc::cu::MergedGeometry> fsc::cu::IndexedGeometry::getBase() const { 
	return cupnp::getPointerField<fsc::cu::DataRef<fsc::cu::MergedGeometry>, 0>(structure, data, reinterpret_cast<const capnp::word*>(BASE_DEFAULT_VALUE));
} 

fsc::cu::CartesianGrid fsc::cu::IndexedGeometry::mutateGrid() { 
	CUPNP_REQUIRE(nonDefaultGrid());
	return cupnp::mutatePointerField<fsc::cu::CartesianGrid, 1>(structure, data);
} 

bool fsc::cu::IndexedGeometry::nonDefaultGrid() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const fsc::cu::CartesianGrid fsc::cu::IndexedGeometry::getGrid() const { 
	return cupnp::getPointerField<fsc::cu::CartesianGrid, 1>(structure, data, reinterpret_cast<const capnp::word*>(GRID_DEFAULT_VALUE));
} 

fsc::cu::ShapedList<cupnp::List<cupnp::List<fsc::cu::IndexedGeometry::ElementRef>>> fsc::cu::IndexedGeometry::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<fsc::cu::ShapedList<cupnp::List<cupnp::List<fsc::cu::IndexedGeometry::ElementRef>>>, 2>(structure, data);
} 

bool fsc::cu::IndexedGeometry::nonDefaultData() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const fsc::cu::ShapedList<cupnp::List<cupnp::List<fsc::cu::IndexedGeometry::ElementRef>>> fsc::cu::IndexedGeometry::getData() const { 
	return cupnp::getPointerField<fsc::cu::ShapedList<cupnp::List<cupnp::List<fsc::cu::IndexedGeometry::ElementRef>>>, 2>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

