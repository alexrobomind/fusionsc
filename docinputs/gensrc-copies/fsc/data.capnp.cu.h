#pragma once 

#include <cupnp/cupnp.h>
#include "data.capnp.h"

namespace fsc{
namespace cu{

template<typename Param0T>
struct DataRef{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	DataRef(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
struct Metadata{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Metadata(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char ID_DEFAULT_VALUE [] = {1, 0, 0, 0, 2, 0, 0, 0};
	inline cupnp::Data mutateId();

	inline const cupnp::Data getId() const;
	inline bool nonDefaultId() const;

	inline uint64_t getTypeId() const;
	inline void setTypeId(uint64_t newVal);

	inline uint64_t getCapTableSize() const;
	inline void setCapTableSize(uint64_t newVal);

	inline uint64_t getDataSize() const;
	inline void setDataSize(uint64_t newVal);

}; // struct typename fsc::cu::DataRef<Param0T>::Metadata

struct Receiver{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	Receiver(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

};

struct DataService{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	DataService(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct Archive{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Archive(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct CapabilityInfo{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline CapabilityInfo(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct DataRefInfo{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline DataRefInfo(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline bool hasNoDataRef() const;
	inline static const unsigned char REF_I_D_DEFAULT_VALUE [] = {1, 0, 0, 0, 2, 0, 0, 0};
	inline cupnp::Data mutateRefID();

	inline const cupnp::Data getRefID() const;
	inline bool nonDefaultRefID() const;
	inline bool hasRefID() const;

}; // struct fsc::cu::Archive::CapabilityInfo::DataRefInfo

	inline fsc::cu::Archive::CapabilityInfo::DataRefInfo getDataRefInfo() const;

}; // struct fsc::cu::Archive::CapabilityInfo

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
	
	inline static const unsigned char ID_DEFAULT_VALUE [] = {1, 0, 0, 0, 2, 0, 0, 0};
	inline cupnp::Data mutateId();

	inline const cupnp::Data getId() const;
	inline bool nonDefaultId() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<cupnp::Data> mutateData();

	inline const cupnp::List<cupnp::Data> getData() const;
	inline bool nonDefaultData() const;

	inline static const unsigned char CAPABILITIES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::Archive::CapabilityInfo> mutateCapabilities();

	inline const cupnp::List<fsc::cu::Archive::CapabilityInfo> getCapabilities() const;
	inline bool nonDefaultCapabilities() const;

	inline uint64_t getTypeId() const;
	inline void setTypeId(uint64_t newVal);

}; // struct fsc::cu::Archive::Entry

	inline static const unsigned char ROOT_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::Archive::Entry mutateRoot();

	inline const fsc::cu::Archive::Entry getRoot() const;
	inline bool nonDefaultRoot() const;

	inline static const unsigned char EXTRA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::Archive::Entry> mutateExtra();

	inline const cupnp::List<fsc::cu::Archive::Entry> getExtra() const;
	inline bool nonDefaultExtra() const;

}; // struct fsc::cu::Archive

struct Float64Tensor{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Float64Tensor(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char SHAPE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateShape();

	inline const cupnp::List<uint64_t> getShape() const;
	inline bool nonDefaultShape() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<double> mutateData();

	inline const cupnp::List<double> getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::Float64Tensor

struct Float32Tensor{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Float32Tensor(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char SHAPE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateShape();

	inline const cupnp::List<uint64_t> getShape() const;
	inline bool nonDefaultShape() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<float> mutateData();

	inline const cupnp::List<float> getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::Float32Tensor

struct Int64Tensor{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Int64Tensor(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char SHAPE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateShape();

	inline const cupnp::List<uint64_t> getShape() const;
	inline bool nonDefaultShape() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<int64_t> mutateData();

	inline const cupnp::List<int64_t> getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::Int64Tensor

struct UInt64Tensor{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline UInt64Tensor(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char SHAPE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateShape();

	inline const cupnp::List<uint64_t> getShape() const;
	inline bool nonDefaultShape() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateData();

	inline const cupnp::List<uint64_t> getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::UInt64Tensor

struct Int32Tensor{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Int32Tensor(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char SHAPE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateShape();

	inline const cupnp::List<uint64_t> getShape() const;
	inline bool nonDefaultShape() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<int32_t> mutateData();

	inline const cupnp::List<int32_t> getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::Int32Tensor

struct UInt32Tensor{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline UInt32Tensor(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char SHAPE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateShape();

	inline const cupnp::List<uint64_t> getShape() const;
	inline bool nonDefaultShape() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint32_t> mutateData();

	inline const cupnp::List<uint32_t> getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::UInt32Tensor

template<typename Param0ListType>
struct ShapedList{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline ShapedList(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char SHAPE_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<uint64_t> mutateShape();

	inline const cupnp::List<uint64_t> getShape() const;
	inline bool nonDefaultShape() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline Param0ListType mutateData();

	inline const Param0ListType getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::cu::ShapedList<Param0ListType>


}}
// ===== struct typename fsc::cu::DataRef<Param0T>::Metadata =====

template<typename Param0T>
cupnp::Data fsc::cu::DataRef<Param0T>::Metadata::mutateId() { 
	CUPNP_REQUIRE(nonDefaultId());
	return cupnp::mutatePointerField<cupnp::Data, 0>(structure, data);
} 

template<typename Param0T>
bool fsc::cu::DataRef<Param0T>::Metadata::nonDefaultId() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

template<typename Param0T>
const cupnp::Data fsc::cu::DataRef<Param0T>::Metadata::getId() const { 
	return cupnp::getPointerField<cupnp::Data, 0>(structure, data, reinterpret_cast<const capnp::word*>(ID_DEFAULT_VALUE));
} 

template<typename Param0T>
void fsc::cu::DataRef<Param0T>::Metadata::setTypeId(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 0>(structure, data, 0, newVal);
} 

template<typename Param0T>
uint64_t fsc::cu::DataRef<Param0T>::Metadata::getTypeId() const { 
	return cupnp::getPrimitiveField<uint64_t, 0>(structure, data, 0);
} 

template<typename Param0T>
void fsc::cu::DataRef<Param0T>::Metadata::setCapTableSize(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 1>(structure, data, 0, newVal);
} 

template<typename Param0T>
uint64_t fsc::cu::DataRef<Param0T>::Metadata::getCapTableSize() const { 
	return cupnp::getPrimitiveField<uint64_t, 1>(structure, data, 0);
} 

template<typename Param0T>
void fsc::cu::DataRef<Param0T>::Metadata::setDataSize(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 2>(structure, data, 0, newVal);
} 

template<typename Param0T>
uint64_t fsc::cu::DataRef<Param0T>::Metadata::getDataSize() const { 
	return cupnp::getPrimitiveField<uint64_t, 2>(structure, data, 0);
} 

// ===== struct fsc::cu::Archive::CapabilityInfo::DataRefInfo =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Archive::CapabilityInfo::DataRefInfo> { using Type = fsc::cu::Archive::CapabilityInfo::DataRefInfo; }; 
} // namespace ::cupnp

bool fsc::cu::Archive::CapabilityInfo::DataRefInfo::hasNoDataRef() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0;
} 

cupnp::Data fsc::cu::Archive::CapabilityInfo::DataRefInfo::mutateRefID() { 
	CUPNP_REQUIRE(nonDefaultRefID());
	return cupnp::mutatePointerField<cupnp::Data, 0>(structure, data);
} 

bool fsc::cu::Archive::CapabilityInfo::DataRefInfo::hasRefID() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1;
} 

bool fsc::cu::Archive::CapabilityInfo::DataRefInfo::nonDefaultRefID() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1 && cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::Data fsc::cu::Archive::CapabilityInfo::DataRefInfo::getRefID() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 1)
		return cupnp::getPointer<cupnp::Data>(reinterpret_cast<const capnp::word*>(REF_I_D_DEFAULT_VALUE));
	
	return cupnp::getPointerField<cupnp::Data, 0>(structure, data, reinterpret_cast<const capnp::word*>(REF_I_D_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::Archive::CapabilityInfo =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Archive::CapabilityInfo> { using Type = fsc::cu::Archive::CapabilityInfo; }; 
} // namespace ::cupnp

fsc::cu::Archive::CapabilityInfo::DataRefInfo fsc::cu::Archive::CapabilityInfo::getDataRefInfo() const { 
	return fsc::cu::Archive::CapabilityInfo::DataRefInfo(structure, data);
} 

// ===== struct fsc::cu::Archive::Entry =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Archive::Entry> { using Type = fsc::cu::Archive::Entry; }; 
} // namespace ::cupnp

cupnp::Data fsc::cu::Archive::Entry::mutateId() { 
	CUPNP_REQUIRE(nonDefaultId());
	return cupnp::mutatePointerField<cupnp::Data, 0>(structure, data);
} 

bool fsc::cu::Archive::Entry::nonDefaultId() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::Data fsc::cu::Archive::Entry::getId() const { 
	return cupnp::getPointerField<cupnp::Data, 0>(structure, data, reinterpret_cast<const capnp::word*>(ID_DEFAULT_VALUE));
} 

cupnp::List<cupnp::Data> fsc::cu::Archive::Entry::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::List<cupnp::Data>, 1>(structure, data);
} 

bool fsc::cu::Archive::Entry::nonDefaultData() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<cupnp::Data> fsc::cu::Archive::Entry::getData() const { 
	return cupnp::getPointerField<cupnp::List<cupnp::Data>, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::Archive::CapabilityInfo> fsc::cu::Archive::Entry::mutateCapabilities() { 
	CUPNP_REQUIRE(nonDefaultCapabilities());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::Archive::CapabilityInfo>, 2>(structure, data);
} 

bool fsc::cu::Archive::Entry::nonDefaultCapabilities() const { 
	return cupnp::hasPointerField<2>(structure, data);
} 

const cupnp::List<fsc::cu::Archive::CapabilityInfo> fsc::cu::Archive::Entry::getCapabilities() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::Archive::CapabilityInfo>, 2>(structure, data, reinterpret_cast<const capnp::word*>(CAPABILITIES_DEFAULT_VALUE));
} 

void fsc::cu::Archive::Entry::setTypeId(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 0>(structure, data, 0, newVal);
} 

uint64_t fsc::cu::Archive::Entry::getTypeId() const { 
	return cupnp::getPrimitiveField<uint64_t, 0>(structure, data, 0);
} 

// ===== struct fsc::cu::Archive =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Archive> { using Type = fsc::cu::Archive; }; 
} // namespace ::cupnp

fsc::cu::Archive::Entry fsc::cu::Archive::mutateRoot() { 
	CUPNP_REQUIRE(nonDefaultRoot());
	return cupnp::mutatePointerField<fsc::cu::Archive::Entry, 0>(structure, data);
} 

bool fsc::cu::Archive::nonDefaultRoot() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const fsc::cu::Archive::Entry fsc::cu::Archive::getRoot() const { 
	return cupnp::getPointerField<fsc::cu::Archive::Entry, 0>(structure, data, reinterpret_cast<const capnp::word*>(ROOT_DEFAULT_VALUE));
} 

cupnp::List<fsc::cu::Archive::Entry> fsc::cu::Archive::mutateExtra() { 
	CUPNP_REQUIRE(nonDefaultExtra());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::Archive::Entry>, 1>(structure, data);
} 

bool fsc::cu::Archive::nonDefaultExtra() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<fsc::cu::Archive::Entry> fsc::cu::Archive::getExtra() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::Archive::Entry>, 1>(structure, data, reinterpret_cast<const capnp::word*>(EXTRA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::Float64Tensor =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Float64Tensor> { using Type = fsc::cu::Float64Tensor; }; 
} // namespace ::cupnp

cupnp::List<uint64_t> fsc::cu::Float64Tensor::mutateShape() { 
	CUPNP_REQUIRE(nonDefaultShape());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

bool fsc::cu::Float64Tensor::nonDefaultShape() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::Float64Tensor::getShape() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(SHAPE_DEFAULT_VALUE));
} 

cupnp::List<double> fsc::cu::Float64Tensor::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::List<double>, 1>(structure, data);
} 

bool fsc::cu::Float64Tensor::nonDefaultData() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<double> fsc::cu::Float64Tensor::getData() const { 
	return cupnp::getPointerField<cupnp::List<double>, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::Float32Tensor =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Float32Tensor> { using Type = fsc::cu::Float32Tensor; }; 
} // namespace ::cupnp

cupnp::List<uint64_t> fsc::cu::Float32Tensor::mutateShape() { 
	CUPNP_REQUIRE(nonDefaultShape());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

bool fsc::cu::Float32Tensor::nonDefaultShape() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::Float32Tensor::getShape() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(SHAPE_DEFAULT_VALUE));
} 

cupnp::List<float> fsc::cu::Float32Tensor::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::List<float>, 1>(structure, data);
} 

bool fsc::cu::Float32Tensor::nonDefaultData() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<float> fsc::cu::Float32Tensor::getData() const { 
	return cupnp::getPointerField<cupnp::List<float>, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::Int64Tensor =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Int64Tensor> { using Type = fsc::cu::Int64Tensor; }; 
} // namespace ::cupnp

cupnp::List<uint64_t> fsc::cu::Int64Tensor::mutateShape() { 
	CUPNP_REQUIRE(nonDefaultShape());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

bool fsc::cu::Int64Tensor::nonDefaultShape() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::Int64Tensor::getShape() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(SHAPE_DEFAULT_VALUE));
} 

cupnp::List<int64_t> fsc::cu::Int64Tensor::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::List<int64_t>, 1>(structure, data);
} 

bool fsc::cu::Int64Tensor::nonDefaultData() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<int64_t> fsc::cu::Int64Tensor::getData() const { 
	return cupnp::getPointerField<cupnp::List<int64_t>, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::UInt64Tensor =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::UInt64Tensor> { using Type = fsc::cu::UInt64Tensor; }; 
} // namespace ::cupnp

cupnp::List<uint64_t> fsc::cu::UInt64Tensor::mutateShape() { 
	CUPNP_REQUIRE(nonDefaultShape());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

bool fsc::cu::UInt64Tensor::nonDefaultShape() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::UInt64Tensor::getShape() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(SHAPE_DEFAULT_VALUE));
} 

cupnp::List<uint64_t> fsc::cu::UInt64Tensor::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 1>(structure, data);
} 

bool fsc::cu::UInt64Tensor::nonDefaultData() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::UInt64Tensor::getData() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::Int32Tensor =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::Int32Tensor> { using Type = fsc::cu::Int32Tensor; }; 
} // namespace ::cupnp

cupnp::List<uint64_t> fsc::cu::Int32Tensor::mutateShape() { 
	CUPNP_REQUIRE(nonDefaultShape());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

bool fsc::cu::Int32Tensor::nonDefaultShape() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::Int32Tensor::getShape() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(SHAPE_DEFAULT_VALUE));
} 

cupnp::List<int32_t> fsc::cu::Int32Tensor::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::List<int32_t>, 1>(structure, data);
} 

bool fsc::cu::Int32Tensor::nonDefaultData() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<int32_t> fsc::cu::Int32Tensor::getData() const { 
	return cupnp::getPointerField<cupnp::List<int32_t>, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::UInt32Tensor =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::UInt32Tensor> { using Type = fsc::cu::UInt32Tensor; }; 
} // namespace ::cupnp

cupnp::List<uint64_t> fsc::cu::UInt32Tensor::mutateShape() { 
	CUPNP_REQUIRE(nonDefaultShape());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

bool fsc::cu::UInt32Tensor::nonDefaultShape() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<uint64_t> fsc::cu::UInt32Tensor::getShape() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(SHAPE_DEFAULT_VALUE));
} 

cupnp::List<uint32_t> fsc::cu::UInt32Tensor::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::List<uint32_t>, 1>(structure, data);
} 

bool fsc::cu::UInt32Tensor::nonDefaultData() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::List<uint32_t> fsc::cu::UInt32Tensor::getData() const { 
	return cupnp::getPointerField<cupnp::List<uint32_t>, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::ShapedList<Param0ListType> =====

// CuFor specializaation
namespace cupnp {
template<typename Param0ListType>
struct CuFor_<fsc::ShapedList<Param0ListType>> { using Type = fsc::cu::ShapedList<Param0ListType>; }; 
} // namespace ::cupnp

template<typename Param0ListType>
cupnp::List<uint64_t> fsc::cu::ShapedList<Param0ListType>::mutateShape() { 
	CUPNP_REQUIRE(nonDefaultShape());
	return cupnp::mutatePointerField<cupnp::List<uint64_t>, 0>(structure, data);
} 

template<typename Param0ListType>
bool fsc::cu::ShapedList<Param0ListType>::nonDefaultShape() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

template<typename Param0ListType>
const cupnp::List<uint64_t> fsc::cu::ShapedList<Param0ListType>::getShape() const { 
	return cupnp::getPointerField<cupnp::List<uint64_t>, 0>(structure, data, reinterpret_cast<const capnp::word*>(SHAPE_DEFAULT_VALUE));
} 

template<typename Param0ListType>
Param0ListType fsc::cu::ShapedList<Param0ListType>::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<Param0ListType, 1>(structure, data);
} 

template<typename Param0ListType>
bool fsc::cu::ShapedList<Param0ListType>::nonDefaultData() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

template<typename Param0ListType>
const Param0ListType fsc::cu::ShapedList<Param0ListType>::getData() const { 
	return cupnp::getPointerField<Param0ListType, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

