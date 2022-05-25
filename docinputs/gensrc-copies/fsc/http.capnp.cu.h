#pragma once 

#include <cupnp/cupnp.h>
#include "http.capnp.h"

namespace fsc{
namespace cu{

struct HttpRoot{
	static constexpr cupnp::Kind kind = cupnp::Kind::STRUCT;
	
	uint64_t structure;
	cupnp::Location data;
	
	inline HttpRoot(uint64_t structure, cupnp::Location data) :
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
	
	inline static const unsigned char LOC_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateLoc();

	inline const cupnp::Text getLoc() const;
	inline bool nonDefaultLoc() const;

	inline static const unsigned char TEXT_DEFAULT_VALUE [] = {1, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::Text mutateText();

	inline const cupnp::Text getText() const;
	inline bool nonDefaultText() const;
	inline bool hasText() const;

	inline static const unsigned char DATA_DEFAULT_VALUE [] = {1, 0, 0, 0, 2, 0, 0, 0};
	inline cupnp::Data mutateData();

	inline const cupnp::Data getData() const;
	inline bool nonDefaultData() const;
	inline bool hasData() const;

}; // struct fsc::cu::HttpRoot::Entry

	inline static const unsigned char ENTRIES_DEFAULT_VALUE [] = {0, 0, 0, 0, 0, 0, 0, 0};
	inline cupnp::List<fsc::cu::HttpRoot::Entry> mutateEntries();

	inline const cupnp::List<fsc::cu::HttpRoot::Entry> getEntries() const;
	inline bool nonDefaultEntries() const;

}; // struct fsc::cu::HttpRoot


}}
// ===== struct fsc::cu::HttpRoot::Entry =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::HttpRoot::Entry> { using Type = fsc::cu::HttpRoot::Entry; }; 
} // namespace ::cupnp

cupnp::Text fsc::cu::HttpRoot::Entry::mutateLoc() { 
	CUPNP_REQUIRE(nonDefaultLoc());
	return cupnp::mutatePointerField<cupnp::Text, 0>(structure, data);
} 

bool fsc::cu::HttpRoot::Entry::nonDefaultLoc() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::Text fsc::cu::HttpRoot::Entry::getLoc() const { 
	return cupnp::getPointerField<cupnp::Text, 0>(structure, data, reinterpret_cast<const capnp::word*>(LOC_DEFAULT_VALUE));
} 

cupnp::Text fsc::cu::HttpRoot::Entry::mutateText() { 
	CUPNP_REQUIRE(nonDefaultText());
	return cupnp::mutatePointerField<cupnp::Text, 1>(structure, data);
} 

bool fsc::cu::HttpRoot::Entry::hasText() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0;
} 

bool fsc::cu::HttpRoot::Entry::nonDefaultText() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 0 && cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::Text fsc::cu::HttpRoot::Entry::getText() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 0)
		return cupnp::getPointer<cupnp::Text>(reinterpret_cast<const capnp::word*>(TEXT_DEFAULT_VALUE));
	
	return cupnp::getPointerField<cupnp::Text, 1>(structure, data, reinterpret_cast<const capnp::word*>(TEXT_DEFAULT_VALUE));
} 

cupnp::Data fsc::cu::HttpRoot::Entry::mutateData() { 
	CUPNP_REQUIRE(nonDefaultData());
	return cupnp::mutatePointerField<cupnp::Data, 1>(structure, data);
} 

bool fsc::cu::HttpRoot::Entry::hasData() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1;
} 

bool fsc::cu::HttpRoot::Entry::nonDefaultData() const { 
	return cupnp::getDiscriminant<0>(structure, data) == 1 && cupnp::hasPointerField<1>(structure, data);
} 

const cupnp::Data fsc::cu::HttpRoot::Entry::getData() const { 
	if(cupnp::getDiscriminant<0>(structure, data) != 1)
		return cupnp::getPointer<cupnp::Data>(reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
	
	return cupnp::getPointerField<cupnp::Data, 1>(structure, data, reinterpret_cast<const capnp::word*>(DATA_DEFAULT_VALUE));
} 

// ===== struct fsc::cu::HttpRoot =====

// CuFor specializaation
namespace cupnp {
template<>
struct CuFor_<fsc::HttpRoot> { using Type = fsc::cu::HttpRoot; }; 
} // namespace ::cupnp

cupnp::List<fsc::cu::HttpRoot::Entry> fsc::cu::HttpRoot::mutateEntries() { 
	CUPNP_REQUIRE(nonDefaultEntries());
	return cupnp::mutatePointerField<cupnp::List<fsc::cu::HttpRoot::Entry>, 0>(structure, data);
} 

bool fsc::cu::HttpRoot::nonDefaultEntries() const { 
	return cupnp::hasPointerField<0>(structure, data);
} 

const cupnp::List<fsc::cu::HttpRoot::Entry> fsc::cu::HttpRoot::getEntries() const { 
	return cupnp::getPointerField<cupnp::List<fsc::cu::HttpRoot::Entry>, 0>(structure, data, reinterpret_cast<const capnp::word*>(ENTRIES_DEFAULT_VALUE));
} 

