#pragma once 

#include <cupnp/cupnp.h>
#include "devices/w7x-test.capnp.h"
#include "../data.capnp.cu.h"
#include "../geometry.capnp.cu.h"
#include "../http.capnp.cu.h"
#include "../magnetics.capnp.cu.h"
#include "w7x.capnp.cu.h"

namespace fsc{
namespace devices{
namespace w7x{
namespace cu{

template<typename Param0T>
struct DBTest{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline DBTest(uint64_t structure, cupnp::Location data) :
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
	
	inline uint64_t getId() const;
	inline void setId(uint64_t newVal);

	inline static const unsigned char RESULT_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline Param0T mutateResult();

	inline const Param0T getResult() const;
	inline bool nonDefaultResult() const;
	inline bool hasResult() const;

	inline bool hasNotFound() const;
}; // struct typename fsc::devices::w7x::cu::DBTest<Param0T>::Entry

	inline static const unsigned char HTTP_ROOT_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline fsc::cu::HttpRoot mutateHttpRoot();

	inline const fsc::cu::HttpRoot getHttpRoot() const;
	inline bool nonDefaultHttpRoot() const;

	inline static const unsigned char ENTRIES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<typename fsc::devices::w7x::cu::DBTest<Param0T>::Entry> mutateEntries();

	inline const cupnp::List<typename fsc::devices::w7x::cu::DBTest<Param0T>::Entry> getEntries() const;
	inline bool nonDefaultEntries() const;

}; // struct fsc::devices::w7x::cu::DBTest<Param0T>


}}}}
// ===== struct typename fsc::devices::w7x::cu::DBTest<Param0T>::Entry =====

template<typename Param0T>
void fsc::devices::w7x::cu::DBTest<Param0T>::Entry::setId(uint64_t newVal) { 
	cupnp::setPrimitiveField<uint64_t, 0>(structure, data, 0, newVal);
} 

template<typename Param0T>
uint64_t fsc::devices::w7x::cu::DBTest<Param0T>::Entry::getId() const { 
	return cupnp::getPrimitiveField<uint64_t, 0>(structure, data, 0);
} 

template<typename Param0T>
Param0T fsc::devices::w7x::cu::DBTest<Param0T>::Entry::mutateResult() { 
	CUPNP_REQUIRE(nonDefaultResult());
	return cupnp::mutatePointerField<Param0T, 0>(structure, data);
} 

template<typename Param0T>
bool fsc::devices::w7x::cu::DBTest<Param0T>::Entry::hasResult() const { 
	return cupnp::getDiscriminant<4>(structure, data) == 0;
} 

template<typename Param0T>
bool fsc::devices::w7x::cu::DBTest<Param0T>::Entry::nonDefaultResult() const { 
	return cupnp::getDiscriminant<4>(structure, data) == 0 && cupnp::hasPointerField<0>(structure, data);
} 

template<typename Param0T>
const Param0T fsc::devices::w7x::cu::DBTest<Param0T>::Entry::getResult() const { 
	if(cupnp::getDiscriminant<4>(structure, data) != 0)
		return cupnp::getPointer<Param0T>(reinterpret_cast<const capnp::word*>(RESULT_DEFAULT_VALUE));
	
	return cupnp::getPointerField<Param0T, 0>(structure, data, reinterpret_cast<const capnp::word*>(RESULT_DEFAULT_VALUE));
} 

template<typename Param0T>
bool fsc::devices::w7x::cu::DBTest<Param0T>::Entry::hasNotFound() const { 
	return cupnp::getDiscriminant<4>(structure, data) == 1;
} 

// ===== struct fsc::devices::w7x::cu::DBTest<Param0T> =====

// CuFor specializaation
namespace cupnp {
template<typename Param0T>
struct CuFor_<fsc::devices::w7x::DBTest<Param0T>> { using Type = fsc::devices::w7x::cu::DBTest<Param0T>; }; 
} // namespace ::cupnp

template<typename Param0T>
fsc::cu::HttpRoot fsc::devices::w7x::cu::DBTest<Param0T>::mutateHttpRoot() { 
	CUPNP_REQUIRE(nonDefaultHttpRoot());
	return cupnp::mutatePointerField<fsc::cu::HttpRoot, 0>(structure, data);
} 

template<typename Param0T>
bool fsc::devices::w7x::cu::DBTest<Param0T>::nonDefaultHttpRoot() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

template<typename Param0T>
const fsc::cu::HttpRoot fsc::devices::w7x::cu::DBTest<Param0T>::getHttpRoot() const { 
	return cupnp::getPointerField<fsc::cu::HttpRoot, 0>(structure, data, reinterpret_cast<const capnp::word*>(HTTP_ROOT_DEFAULT_VALUE));
} 

template<typename Param0T>
cupnp::List<typename fsc::devices::w7x::cu::DBTest<Param0T>::Entry> fsc::devices::w7x::cu::DBTest<Param0T>::mutateEntries() { 
	CUPNP_REQUIRE(nonDefaultEntries());
	return cupnp::mutatePointerField<cupnp::List<typename fsc::devices::w7x::cu::DBTest<Param0T>::Entry>, 1>(structure, data);
} 

template<typename Param0T>
bool fsc::devices::w7x::cu::DBTest<Param0T>::nonDefaultEntries() const { 
	return cupnp::hasPointerField<1>(structure, data);
} 

template<typename Param0T>
const cupnp::List<typename fsc::devices::w7x::cu::DBTest<Param0T>::Entry> fsc::devices::w7x::cu::DBTest<Param0T>::getEntries() const { 
	return cupnp::getPointerField<cupnp::List<typename fsc::devices::w7x::cu::DBTest<Param0T>::Entry>, 1>(structure, data, reinterpret_cast<const capnp::word*>(ENTRIES_DEFAULT_VALUE));
} 

