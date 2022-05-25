#pragma once 

#include <cupnp/cupnp.h>
#include "data-test.capnp.h"
#include "data.capnp.cu.h"

namespace fsc{
namespace test{
namespace cu{

struct DataHolder{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline DataHolder(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char DATA_DEFAULT_VALUE [] = {1, 0, 0, 0, 2, 0, 0, 0};
	inline cupnp::Data mutateData();

	inline const cupnp::Data getData() const;
	inline bool nonDefaultData() const;

}; // struct fsc::test::cu::DataHolder

template<typename Param0T>
struct DataRefHolder{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline DataRefHolder(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char REF_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::DataRef<Param0T> mutateRef();

	inline const fsc::cu::DataRef<Param0T> getRef() const;
	inline bool nonDefaultRef() const;

}; // struct fsc::test::cu::DataRefHolder<Param0T>

struct A{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	A(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct B{
	static constexpr cupnp::Kind kind = cupnp::Kind::INTERFACE;
	
	// Interface pointer that holds the capability table offset
	uint64_t ptrData;
	
	B(uint64_t structure, cupnp::Location data) : ptrData(structure) {
		cupnp::validateInterfacePointer(structure, data);
	}
	
	bool isDefault() { return ptrData == 0; }
	
};

struct TestStruct{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline TestStruct(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
struct Ints{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Ints(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline int8_t getInt8() const;
	inline void setInt8(int8_t newVal);

	inline bool hasInt8() const;
	inline int16_t getInt16() const;
	inline void setInt16(int16_t newVal);

	inline bool hasInt16() const;
	inline int32_t getInt32() const;
	inline void setInt32(int32_t newVal);

	inline bool hasInt32() const;
	inline int64_t getInt64() const;
	inline void setInt64(int64_t newVal);

	inline bool hasInt64() const;
}; // struct fsc::test::cu::TestStruct::Ints

struct Uints{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Uints(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline uint8_t getUint8() const;
	inline void setUint8(uint8_t newVal);

	inline bool hasUint8() const;
	inline uint16_t getUint16() const;
	inline void setUint16(uint16_t newVal);

	inline bool hasUint16() const;
	inline uint32_t getUint32() const;
	inline void setUint32(uint32_t newVal);

	inline bool hasUint32() const;
	inline uint64_t getUint64() const;
	inline void setUint64(uint64_t newVal);

	inline bool hasUint64() const;
}; // struct fsc::test::cu::TestStruct::Uints

struct Pointers{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline Pointers(uint64_t structure, cupnp::Location data) :
		structure(structure),
		data(data)
	{
		cupnp::validateStructPointer(structure, data);
	}
	
	inline bool isDefault() { return structure == 0; }
	
	inline static const unsigned char DATA_DEFAULT_VALUE [] = {1, 0, 0, 0, 2, 0, 0, 0};
	inline cupnp::Data mutateData();

	inline const cupnp::Data getData() const;
	inline bool nonDefaultData() const;

	inline static const unsigned char TEXT_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateText();

	inline const cupnp::Text getText() const;
	inline bool nonDefaultText() const;

}; // struct fsc::test::cu::TestStruct::Pointers

	inline bool getBool() const;
	inline void setBool(bool newVal);

	inline fsc::test::cu::TestStruct::Ints getInts() const;

	inline fsc::test::cu::TestStruct::Uints getUints() const;

	inline fsc::test::cu::TestStruct::Pointers getPointers() const;

}; // struct fsc::test::cu::TestStruct


}}}
// ===== struct fsc::test::cu::DataHolder =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::test::DataHolder> { using Type = fsc::test::cu::DataHolder; }; 
} // namespace ::cupnp

cupnp::Data fsc::test::cu::DataHolder::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::Data, 0>(structure, data);
} 

bool fsc::test::cu::DataHolder::nonDefaultData() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::Data fsc::test::cu::DataHolder::getData() const { 
	return cupnp::getPointerField<cupnp::Data, 0>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::test::cu::DataRefHolder<Param0T> =====

// CuFor specializaation
namespace cupnp {
template<typename Param0T>
struct CuFor_<fsc::test::DataRefHolder<Param0T>> { using Type = fsc::test::cu::DataRefHolder<Param0T>; }; 
} // namespace ::cupnp

template<typename Param0T>
fsc::cu::DataRef<Param0T> fsc::test::cu::DataRefHolder<Param0T>::mutateRef() { 
	CUPNP_REQUIRE(nonDefaultRef());
	return cupnp::mutatePointerField<fsc::cu::DataRef<Param0T>, 0>(structure, data);
} 

template<typename Param0T>
bool fsc::test::cu::DataRefHolder<Param0T>::nonDefaultRef() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

template<typename Param0T>
const fsc::cu::DataRef<Param0T> fsc::test::cu::DataRefHolder<Param0T>::getRef() const { 
	return cupnp::getPointerField<fsc::cu::DataRef<Param0T>, 0>(structure, data, reinterpret_cast<const capnp::word*>(REF_DEFAULT_VALUE));
} 

// ===== struct fsc::test::cu::TestStruct::Ints =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::test::TestStruct::Ints> { using Type = fsc::test::cu::TestStruct::Ints; }; 
} // namespace ::cupnp

void fsc::test::cu::TestStruct::Ints::setInt8(int8_t newVal) { 
	cupnp::setPrimitiveField<int8_t, 1>(structure, data, 0, newVal);
	cupnp::setDiscriminant<1>(structure, data, 0);
} 

int8_t fsc::test::cu::TestStruct::Ints::getInt8() const { 
	if(cupnp::getDiscriminant<1>(structure, data) != 0)
		return 0;
	
	return cupnp::getPrimitiveField<int8_t, 1>(structure, data, 0);
} 

bool fsc::test::cu::TestStruct::Ints::hasInt8() const { 
	return cupnp::getDiscriminant<1>(structure, data) == 0;
} 

void fsc::test::cu::TestStruct::Ints::setInt16(int16_t newVal) { 
	cupnp::setPrimitiveField<int16_t, 2>(structure, data, 0, newVal);
	cupnp::setDiscriminant<1>(structure, data, 1);
} 

int16_t fsc::test::cu::TestStruct::Ints::getInt16() const { 
	if(cupnp::getDiscriminant<1>(structure, data) != 1)
		return 0;
	
	return cupnp::getPrimitiveField<int16_t, 2>(structure, data, 0);
} 

bool fsc::test::cu::TestStruct::Ints::hasInt16() const { 
	return cupnp::getDiscriminant<1>(structure, data) == 1;
} 

void fsc::test::cu::TestStruct::Ints::setInt32(int32_t newVal) { 
	cupnp::setPrimitiveField<int32_t, 1>(structure, data, 0, newVal);
	cupnp::setDiscriminant<1>(structure, data, 2);
} 

int32_t fsc::test::cu::TestStruct::Ints::getInt32() const { 
	if(cupnp::getDiscriminant<1>(structure, data) != 2)
		return 0;
	
	return cupnp::getPrimitiveField<int32_t, 1>(structure, data, 0);
} 

bool fsc::test::cu::TestStruct::Ints::hasInt32() const { 
	return cupnp::getDiscriminant<1>(structure, data) == 2;
} 

void fsc::test::cu::TestStruct::Ints::setInt64(int64_t newVal) { 
	cupnp::setPrimitiveField<int64_t, 1>(structure, data, 0, newVal);
	cupnp::setDiscriminant<1>(structure, data, 3);
} 

int64_t fsc::test::cu::TestStruct::Ints::getInt64() const { 
	if(cupnp::getDiscriminant<1>(structure, data) != 3)
		return 0;
	
	return cupnp::getPrimitiveField<int64_t, 1>(structure, data, 0);
} 

bool fsc::test::cu::TestStruct::Ints::hasInt64() const { 
	return cupnp::getDiscriminant<1>(structure, data) == 3;
} 

// ===== struct fsc::test::cu::TestStruct::Uints =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::test::TestStruct::Uints> { using Type = fsc::test::cu::TestStruct::Uints; }; 
} // namespace ::cupnp

void fsc::test::cu::TestStruct::Uints::setUint8(uint8_t newVal) { 
	cupnp::setPrimitiveField<uint8_t, 16>(structure, data, 0, newVal);
	cupnp::setDiscriminant<9>(structure, data, 0);
} 

uint8_t fsc::test::cu::TestStruct::Uints::getUint8() const { 
	if(cupnp::getDiscriminant<9>(structure, data) != 0)
		return 0;
	
	return cupnp::getPrimitiveField<uint8_t, 16>(structure, data, 0);
} 

bool fsc::test::cu::TestStruct::Uints::hasUint8() const { 
	return cupnp::getDiscriminant<9>(structure, data) == 0;
} 

void fsc::test::cu::TestStruct::Uints::setUint16(uint16_t newVal) { 
	cupnp::setPrimitiveField<uint16_t, 8>(structure, data, 0, newVal);
	cupnp::setDiscriminant<9>(structure, data, 1);
} 

uint16_t fsc::test::cu::TestStruct::Uints::getUint16() const { 
	if(cupnp::getDiscriminant<9>(structure, data) != 1)
		return 0;
	
	return cupnp::getPrimitiveField<uint16_t, 8>(structure, data, 0);
} 

bool fsc::test::cu::TestStruct::Uints::hasUint16() const { 
	return cupnp::getDiscriminant<9>(structure, data) == 1;
} 

void fsc::test::cu::TestStruct::Uints::setUint32(uint32_t newVal) { 
	cupnp::setPrimitiveField<uint32_t, 5>(structure, data, 0, newVal);
	cupnp::setDiscriminant<9>(structure, data, 2);
} 

uint32_t fsc::test::cu::TestStruct::Uints::getUint32() const { 
	if(cupnp::getDiscriminant<9>(structure, data) != 2)
		return 0;
	
	return cupnp::getPrimitiveField<uint32_t, 5>(structure, data, 0);
} 

bool fsc::test::cu::TestStruct::Uints::hasUint32() const { 
	return cupnp::getDiscriminant<9>(structure, data) == 2;
} 

void fsc::test::cu::TestStruct::Uints::setUint64(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 3>(structure, data, 0, newVal);
	cupnp::setDiscriminant<9>(structure, data, 3);
} 

uint64_t fsc::test::cu::TestStruct::Uints::getUint64() const { 
	if(cupnp::getDiscriminant<9>(structure, data) != 3)
		return 0;
	
	return cupnp::getPrimitiveField<uint64_t, 3>(structure, data, 0);
} 

bool fsc::test::cu::TestStruct::Uints::hasUint64() const { 
	return cupnp::getDiscriminant<9>(structure, data) == 3;
} 

// ===== struct fsc::test::cu::TestStruct::Pointers =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::test::TestStruct::Pointers> { using Type = fsc::test::cu::TestStruct::Pointers; }; 
} // namespace ::cupnp

cupnp::Data fsc::test::cu::TestStruct::Pointers::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::Data, 0>(structure, data);
} 

bool fsc::test::cu::TestStruct::Pointers::nonDefaultData() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::Data fsc::test::cu::TestStruct::Pointers::getData() const { 
	return cupnp::getPointerField<cupnp::Data, 0>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

cupnp::Text fsc::test::cu::TestStruct::Pointers::mutateText() { 
	CUPNP_REQUIRE(nonDefaultText());
	return cupnp::mutatePointerField<cupnp::Text, 1>(structure, data);
} 

bool fsc::test::cu::TestStruct::Pointers::nonDefaultText() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::Text fsc::test::cu::TestStruct::Pointers::getText() const { 
	return cupnp::getPointerField<cupnp::Text, 1>(structure, data, reinterpret_cast<const capnp::word*>(TEXT_DEFAULT_VALUE));
} 

// ===== struct fsc::test::cu::TestStruct =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::test::TestStruct> { using Type = fsc::test::cu::TestStruct; }; 
} // namespace ::cupnp

void fsc::test::cu::TestStruct::setBool(bool newVal) { 
	cupnp::setPrimitiveField<bool, 0>(structure, data, false, newVal);
} 

bool fsc::test::cu::TestStruct::getBool() const { 
	return cupnp::getPrimitiveField<bool, 0>(structure, data, false);
} 

fsc::test::cu::TestStruct::Ints fsc::test::cu::TestStruct::getInts() const { 
	return fsc::test::cu::TestStruct::Ints(structure, data);
} 

fsc::test::cu::TestStruct::Uints fsc::test::cu::TestStruct::getUints() const { 
	return fsc::test::cu::TestStruct::Uints(structure, data);
} 

fsc::test::cu::TestStruct::Pointers fsc::test::cu::TestStruct::getPointers() const { 
	return fsc::test::cu::TestStruct::Pointers(structure, data);
} 

