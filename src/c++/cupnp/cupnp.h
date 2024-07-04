#include <capnp/common.h>
#include <capnp/list.h>
#include <capnp/message.h>

#include <kj/common.h>

#include <algorithm>

#include "gpu_defs.h"

#pragma once

namespace cupnp {

enum class Kind {
	PRIMITIVE,
	BLOB,
	LIST,
	STRUCT,
	ANY_POINTER,
	INTERFACE,
	ENUM
};

namespace internal {
	template<typename T>
	struct KindFor_ { static inline constexpr Kind kind = T::kind; };
	
	#define DECLARE_KIND(TYPE, VALUE) \
		template<> \
		struct KindFor_<TYPE> { static inline constexpr Kind kind = Kind::VALUE; }
	
	DECLARE_KIND(uint8_t, PRIMITIVE);
	DECLARE_KIND(int8_t, PRIMITIVE);
	DECLARE_KIND(uint16_t, PRIMITIVE);
	DECLARE_KIND(int16_t, PRIMITIVE);
	DECLARE_KIND(uint32_t, PRIMITIVE);
	DECLARE_KIND(int32_t, PRIMITIVE);
	DECLARE_KIND(uint64_t, PRIMITIVE);
	DECLARE_KIND(int64_t, PRIMITIVE);
	DECLARE_KIND(bool, PRIMITIVE);
	DECLARE_KIND(float, PRIMITIVE);
	DECLARE_KIND(double, PRIMITIVE);
	
	#undef DECLARE_KIND
}

template<typename T>
constexpr Kind KindFor = internal::KindFor_<T>::kind;
	
template<typename T>
struct CuFor_ { using Type = T; };

template<typename T>
using CuFor = typename CuFor_<T>::Type;

struct SegmentTable {
	struct Entry {
		inline CUPNP_FUNCTION capnp::word* begin() { return entry; }
		inline CUPNP_FUNCTION capnp::word* end() { return entry + sizeInWords; }
		inline CUPNP_FUNCTION size_t size() { return sizeInWords; }
		
		Entry(kj::ArrayPtr<capnp::word> src) :
			entry(src.begin()), sizeInWords(src.size())
		{}
		
	private:
		capnp::word* entry;
		size_t sizeInWords;
	};
	
	SegmentTable(kj::ArrayPtr<Entry> src) :
		entries(src.begin()),
		_size(src.size())
	{}
			
	inline CUPNP_FUNCTION Entry* begin() { return entries; }
	inline CUPNP_FUNCTION Entry* end() { return entries + _size; }
	inline CUPNP_FUNCTION size_t size() { return _size; }
	
	inline CUPNP_FUNCTION SegmentTable() = default;
	inline CUPNP_FUNCTION SegmentTable(std::nullptr_t) :
		entries(nullptr), _size(0)
	{}
	
	inline CUPNP_FUNCTION bool operator==(std::nullptr_t) { return entries == nullptr; }
	inline CUPNP_FUNCTION bool operator!=(std::nullptr_t) { return entries != nullptr; }
	
	inline CUPNP_FUNCTION Entry& operator[](size_t i) { return entries[i]; }
	inline CUPNP_FUNCTION const Entry& operator[](size_t i) const { return entries[i]; }
	
private:
	Entry* entries;
	size_t _size;
};
	
struct Location {
	unsigned int segmentId;
	unsigned char* ptr;
	
	// kj::ArrayPtr<kj::ArrayPtr<capnp::word>> segments;
	SegmentTable segments;
	
	inline CUPNP_FUNCTION Location() = default;
	inline CUPNP_FUNCTION Location(const capnp::word* message) :
		segmentId(0),
		ptr(reinterpret_cast<unsigned char*>(const_cast<capnp::word*>(message))),
		segments(nullptr)
	{}
	
	template<typename T>
	CUPNP_FUNCTION T read() const {
		// Assume GPU is little-endian
		# ifdef CUPNP_GPUCC
			return *(reinterpret_cast<T*>(ptr));
		# else
			return reinterpret_cast<capnp::_::WireValue<T>*>(ptr)->get();
		# endif
	}
	
	template<typename T>
	CUPNP_FUNCTION void write(T newVal) {
		// Assume GPU is little-endian
		# ifdef CUPNP_GPUCC
			*(reinterpret_cast<T*>(ptr)) = newVal;
		# else
			reinterpret_cast<capnp::_::WireValue<T>*>(ptr)->set(newVal);
		# endif
	}
	
	inline CUPNP_FUNCTION Location operator+(int32_t shift) const {
		Location l2;
		l2.segmentId = segmentId;
		l2.ptr = ptr + shift;
		l2.segments = segments;
		
		return l2;
	}
	
	inline CUPNP_FUNCTION bool isValid(size_t size) {
		if(ptr == nullptr)
			return size == 0;
		
		if(segments == nullptr)
			return true;
		
		// KJ_DBG("Validity check:", segmentId, segments.size());
		
		if(segmentId >= segments.size())
			return false;
		
		auto start = reinterpret_cast<unsigned char*>(/*segments->segments[2 * segmentId]*/segments[segmentId].begin());
		auto end   = reinterpret_cast<unsigned char*>(/*segments->segments[2 * segmentId + 1]*/segments[segmentId].end());
		
		// KJ_DBG("Validity check:", ptr, size, ptr + size, start, end);
		
		if(ptr < start)
			return false;
		
		return ptr + size <= end;
	}
};

inline Location messageRoot(SegmentTable::Entry firstSegment, SegmentTable segmentRefs);

template<typename T>
CUPNP_FUNCTION T getPointer(Location base);

template<typename T>
inline kj::Array<size_t> calculateSizes(const kj::ArrayPtr<T>& segments) {
	auto sizes = kj::heapArrayBuilder<size_t>(segments.size());
	for(const auto& segment : segments)
		sizes.add(segment.size());
	
	return sizes.finish();
}

template<typename T1, typename T2>
inline auto deviceMemcpyAll(T1& dst, const T2& src) {
	CUPNP_REQUIRE(dst.size() == src.size()); 
	
	for(size_t i = 0; i < dst.size(); ++i) { 
		auto err = deviceMemcpy(dst[i].asBytes(), src[i].asBytes()); 
		
		if(err != 0) 
			return err; 
	}
	
	return 0;
}

/**
 * Returns the pointer tag, which is stored in its
 * 2 least significant bits.
 */
inline CUPNP_FUNCTION unsigned char ptrTag(Location in) {
	uint64_t nativeVal = in.read<uint64_t>();
	
	return nativeVal & 3u;
}

/**
 * Given a location pointing to an intra-segment pointer,
 * returns the location pointed at by the pointer.
 */
inline CUPNP_FUNCTION Location decodeNearPtr(Location in) {
	/*
	lsb                      near pointer                       msb
	+-+-----------------------------+-------------------------------+
	|A|             B               |               C               |
	+-+-----------------------------+-------------------------------+

	A (2 bits) = 0, to indicate that this is a struct pointer, or
		1 to indicate that it is a list pointer.
	B (30 bits) = Offset, in words, from the end (!) of the pointer to the
		start of the struct's / list's data section.  Signed (!).
	C (32 bits) = Pointer specific content information.
	*/
	uint64_t nativeVal = in.read<uint64_t>();
	
	uint64_t offsetMask = (1ull << 32) - 1;
	uint64_t ptrTag = nativeVal & 3u;
	
	CUPNP_REQUIRE(ptrTag <= 1);
	
	uint32_t unsignedOffset = (nativeVal & offsetMask) >> 2;
	
	// Sign-extend offset to 32bit number
	// See below link for source:
	// http://graphics.stanford.edu/~seander/bithacks.html#VariableSignExtend
	uint32_t offsetSignMask = (1ul << 29); // Places a 1 at the 30th bit, which is the sign bit
	int32_t offset = (unsignedOffset ^ offsetSignMask) - offsetSignMask;
	
	return in + (offset + 1) * sizeof(capnp::word);
}

/**
 * Given a location by an inter-segment point, stores the location of
 * its landing pad in the second argument and returns the
 * type of landing pad.
 */
inline CUPNP_FUNCTION unsigned char decodeFarPtr(const Location& in, Location& out) {
	/*
	
	lsb                        far pointer                        msb
	+-+-+---------------------------+-------------------------------+
	|A|B|            C              |               D               |
	+-+-+---------------------------+-------------------------------+

	A (2 bits) = 2, to indicate that this is a far pointer.
	B (1 bit) = 0 if the landing pad is one word, 1 if it is two words.
		See explanation below.
	C (29 bits) = Offset, in words, from the start of the target segment
		to the location of the far-pointer landing-pad within that
		segment.  Unsigned.
	D (32 bits) = ID of the target segment.  (Segments are numbered
		sequentially starting from zero.)
		
	If B == 0, then the “landing pad” of a far pointer is normally just
	another pointer, which in turn points to the actual object.

	If B == 1, then the “landing pad” is itself another far pointer that
	is interpreted differently: This far pointer (which always has B = 0)
	points to the start of the object’s content, located in some other
	segment. The landing pad is itself immediately followed by a tag word.
	The tag word looks exactly like an intra-segment pointer to the target
	object would look, except that the offset is always zero.
	
	*/
	uint64_t nativeVal = in.read<uint64_t>();
	
	uint8_t ptrTagVal = nativeVal & 3u;  // A
	CUPNP_REQUIRE(ptrTagVal == 2);
	
	auto message = in.segments;
	CUPNP_REQUIRE(message != nullptr) { out = Location(nullptr); return 0; }
	
	// Far pointer decoding
	uint8_t landingPadType;
	uint32_t landingPadOffset;
	uint32_t segmentId;
	
	landingPadType = (nativeVal >> 2) & 1; // B
	landingPadOffset = (nativeVal & ((1ull << 32) - 1)) >> 3; // C
	segmentId = nativeVal >> 32; // D
	
	CUPNP_REQUIRE(segmentId < message.size()) { out = Location(nullptr); return 0; }
	out.ptr = reinterpret_cast<unsigned char*>(message[segmentId].begin() + landingPadOffset); // Offset calculation is in 8-byte words
	CUPNP_REQUIRE(out.ptr < reinterpret_cast<unsigned char*>(message[segmentId].end())) { out = Location(nullptr); return 0; }
	
	out.segmentId = segmentId;
	out.segments = message;
	
	return landingPadType;
}

template<typename T, unsigned int size = sizeof(T)> struct EncodedType_ {};
template<typename T> struct EncodedType_<T, 1> { using Type = uint8_t; };
template<typename T> struct EncodedType_<T, 2> { using Type = uint16_t; };
template<typename T> struct EncodedType_<T, 4> { using Type = uint32_t; };
template<typename T> struct EncodedType_<T, 8> { using Type = uint64_t; };

template<typename T>
using EncodedType = typename EncodedType_<T>::Type;

template<typename T>
CUPNP_FUNCTION EncodedType<T> encodePrimitive(T value) {
	EncodedType<T> result;
	memcpy(&result, &value, sizeof(T));
	return result;
}

template<typename T>
CUPNP_FUNCTION T decodePrimitive(EncodedType<T> value) {
	T result;
	memcpy(&result, &value, sizeof(T));
	return result;
}

/**
 * Reads a primitive value from the wire data.
 */
template<typename T, uint32_t offset>
CUPNP_FUNCTION const T getPrimitiveField(uint32_t dataSectionSize, Location data, T defaultValue) {
	// uint16_t dataSectionSizeInWords = structure >> 32;
	
	// If the data section can not fully hold this value, retrn the default value instead
	if(sizeof(T) * (offset + 1) > dataSectionSize) {
		return defaultValue;
	}
	
	// Read the bitwise representation from the wire
	EncodedType<T> wireData = (data + offset * sizeof(T)).read<EncodedType<T>>();
	
	// XOR the bitwise representation with the default value
	EncodedType<T> encodedDefault = encodePrimitive(defaultValue);
	wireData ^= encodedDefault;

	// Perform a bitcast
	return decodePrimitive<T>(wireData);
}

template<typename T, uint32_t offset>
CUPNP_FUNCTION const T getPrimitiveField(uint32_t dataSectionSize, Location data) {
	// uint16_t dataSectionSizeInWords = structure >> 32;
	
	// If the data section can not fully hold this value, retrn the default value instead
	if(sizeof(T) * (offset + 1) > dataSectionSize) {
		return 0;
	}
	
	// Read the bitwise representation from the wire
	EncodedType<T> wireData = (data + offset * sizeof(T)).read<EncodedType<T>>();

	// Perform a bitcast
	return decodePrimitive<T>(wireData);
}

template<typename T, uint32_t offset>
CUPNP_FUNCTION void setPrimitiveField(uint32_t dataSectionSize, Location data, T defaultValue, T value) {
	// uint16_t dataSectionSizeInWords = structure >> 32;
	
	CUPNP_REQUIRE(sizeof(T) * (offset + 1) <= dataSectionSize) { return; }
	
	EncodedType<T> wireData       = encodePrimitive(value);
	
	// XOR the bitwise representation with the default value
	EncodedType<T> encodedDefault = encodePrimitive(defaultValue);
	wireData ^= encodedDefault;
	
	(data + offset * sizeof(T)).write(wireData);
}

template<typename T, uint32_t offset>
CUPNP_FUNCTION void setPrimitiveField(uint32_t dataSectionSize, Location data, T value) {
	// uint16_t dataSectionSizeInWords = structure >> 32;
	
	CUPNP_REQUIRE(sizeof(T) * (offset + 1) <= dataSectionSize) { return; }
	
	EncodedType<T> wireData       = encodePrimitive(value);
	(data + offset * sizeof(T)).write(wireData);
}

template<uint32_t offset>
CUPNP_FUNCTION bool getBoolField(uint32_t dataSectionSize, Location data, bool defaultValue) {
	constexpr uint32_t byteOffset = offset >> 3;
	constexpr uint8_t  bitOffset  = offset & 7u;
	constexpr uint8_t  bitMask    = 1u >> bitOffset;
	
	// uint16_t dataSectionSizeInWords = structure >> 32;
	
	if(byteOffset + 1 > dataSectionSize)
		return defaultValue;
	
	uint8_t wireData = (data + byteOffset).read<uint8_t>();
	bool wireBool = wireData & bitMask;
	
	return wireBool != defaultValue;
}
	

template<uint32_t offset>
CUPNP_FUNCTION bool getBoolField(uint32_t dataSectionSize, Location data) {
	constexpr uint32_t byteOffset = offset >> 3;
	constexpr uint8_t  bitOffset  = offset & 7u;
	constexpr uint8_t  bitMask    = 1u >> bitOffset;
	
	// uint16_t dataSectionSizeInWords = structure >> 32;
	
	if(byteOffset + 1 > dataSectionSize)
		return false;
	
	uint8_t wireData = (data + byteOffset).read<uint8_t>();
	bool wireBool = wireData & bitMask;
	
	return wireBool;
}

template<uint32_t offset>
CUPNP_FUNCTION void setBoolField(uint32_t dataSectionSize, Location data, bool defaultValue, bool value) {
	constexpr uint32_t byteOffset = offset >> 3;
	constexpr uint8_t  bitOffset  = offset & 7u;
	constexpr uint8_t  bitMask    = 1u >> bitOffset;
	
	// uint16_t dataSectionSizeInWords = structure >> 32;
	
	CUPNP_REQUIRE(byteOffset + 1 <= dataSectionSize) { return; }
	
	uint8_t wireData = (data + byteOffset).read<uint8_t>();
	
	if(value != defaultValue) {
		wireData |= bitMask;
	} else {
		wireData &= ~bitMask;
	}
	
	(data + byteOffset).write(wireData);
}

template<uint32_t offset>
CUPNP_FUNCTION void setBoolField(uint32_t dataSectionSize, Location data, bool value) {
	constexpr uint32_t byteOffset = offset >> 3;
	constexpr uint8_t  bitOffset  = offset & 7u;
	constexpr uint8_t  bitMask    = 1u >> bitOffset;
	
	// uint16_t dataSectionSizeInWords = structure >> 32;
	
	CUPNP_REQUIRE(byteOffset + 1 <= dataSectionSize) { return; }
	
	uint8_t wireData = (data + byteOffset).read<uint8_t>();
	
	if(value) {
		wireData |= bitMask;
	} else {
		wireData &= ~bitMask;
	}
	
	(data + byteOffset).write(wireData);
}

template<uint32_t offset>
CUPNP_FUNCTION const uint16_t getDiscriminant(uint32_t dataSectionSize, Location data) {
	return getPrimitiveField<uint16_t, offset>(dataSectionSize, data);
}

template<uint32_t offset>
CUPNP_FUNCTION const void setDiscriminant(uint32_t dataSectionSize, Location data, uint16_t newVal) {
	setPrimitiveField<uint16_t, offset>(dataSectionSize, data, newVal);
}

/**
 * Returns a mutable reference to the data in the given pointer field.
 * Fails if the field is out-of-bounds or 0, as the default value storage
 * is not mutable.
 */
template<typename T, uint32_t offset>
CUPNP_FUNCTION typename T::Builder mutatePointerField(uint32_t dataSectionSize, uint16_t pointerSectionSize, Location data) {
	// uint16_t dataSectionSizeInWords = structure >> 32;
	// uint16_t pointerSectionSize = structure >> 48;
	
	CUPNP_REQUIRE(offset < pointerSectionSize) { return getPointer<typename T::Builder>(nullptr); }
	
	Location ptrLoc = data + dataSectionSize + sizeof(capnp::word) * offset;
	bool isDefault = ptrLoc.read<uint64_t>() == 0;
	
	CUPNP_REQUIRE(!isDefault);
	return getPointer<typename T::Builder>(ptrLoc);
}

template<uint32_t offset>
CUPNP_FUNCTION bool hasPointerField(uint32_t dataSectionSize, uint16_t pointerSectionSize, Location data) {
	// uint16_t dataSectionSizeInWords = structure >> 32;
	// uint16_t pointerSectionSize = structure >> 48;
	
	if(offset >= pointerSectionSize)
		return false;
	
	Location ptrLoc = data + dataSectionSize + sizeof(capnp::word) * offset;
	bool isDefault = ptrLoc.read<uint64_t>() == 0;
	
	return !isDefault;
}

/**
 * Creates a value holder of the specified type from a single-segment capnproto message.
 * The memory contents pointed to by data must be a single near-pointer (can be list or structure pointer),
 * which points to the contents of the default value data.
 */
template<typename T>
const CUPNP_FUNCTION typename T::Reader getDefaultValue(const capnp::word* data) {
	// Default value
	Location structureLoc;
	structureLoc.ptr = reinterpret_cast<unsigned char*>(const_cast<capnp::word*>(data));
	structureLoc.segments = nullptr;
	structureLoc.segmentId = -1;
	
	Location dataLoc = decodeNearPtr(structureLoc);
	
	return typename T::Reader(structureLoc.read<uint64_t>(), dataLoc);
}

/**
 * Reads a pointer-type field of a structure in read-only mode. If the pointer is 0
 * (or out of bounds of the structure), the default value will be returned instread.
 */
template<typename T, uint32_t offset>
const CUPNP_FUNCTION typename T::Reader getPointerField(uint32_t dataSectionSize, uint16_t pointerSectionSize, Location data, const capnp::word* defaultValue) {
	//uint16_t dataSectionSizeInWords = structure >> 32;
	//uint16_t pointerSectionSize = structure >> 48;
	
	if(offset >= pointerSectionSize) {
		return getDefaultValue<T>(defaultValue);
	}
	
	Location ptrLoc = data + dataSectionSize + sizeof(capnp::word) * (offset);

	bool isDefault = ptrLoc.read<uint64_t>() == 0;
	if(isDefault) {
		return getDefaultValue<T>(defaultValue);
	}
	
	return getPointer<typename T::Reader>(ptrLoc);
}

/**
 * Compute the size (in 8-byte words) of the data region pointed to by the
 * given data with the provided structure information.
 *
 * Note: Currently only accepts structure pointers.
 */
CUPNP_FUNCTION inline uint32_t computeContentSize(uint64_t structure, Location data) {
	uint8_t tag = structure & 3u;
	if(tag == 0) {
		uint16_t dataSectionSizeInWords = structure >> 32;
		uint16_t pointerSectionSize = structure >> 48;
		return dataSectionSizeInWords + pointerSectionSize;
	} else {
		CUPNP_FAIL_REQUIRE("Implement missing nested list size")
		; // This is on a new line to silence an empty for loop warning from CLANG
		
		#ifndef CUPNP_GPUCC
			KJ_UNREACHABLE;
		#else
			return 0;
		#endif
	}
}

//! Computes the size of a struct's data section
CUPNP_FUNCTION inline uint16_t getDataSectionSizeInWords(uint64_t structureWord) {
	uint16_t dataSectionSizeInWords = structureWord >> 32;
	return dataSectionSizeInWords;
}

//! Computes the size of a struct's data section
CUPNP_FUNCTION inline uint16_t getPointerSectionSize(uint64_t structureWord) {
	uint16_t pointerSectionSize = structureWord >> 48;
	return pointerSectionSize;
}

//! Computes the size of a struct's data section
CUPNP_FUNCTION inline uint32_t getDataSectionSizeInBytes(uint64_t structureWord) {
	uint16_t dataSectionSizeInWords = structureWord >> 32;
	return dataSectionSizeInWords * ((uint32_t) 8);
}

//! Copies data from struct src to struct dst
template<typename T1, typename T2>
CUPNP_FUNCTION inline void copyData(T1& dst, const T2& src) {
	/*uint32_t dsSize1 = getDataSectionSizeInBytes(dst.structure);
	uint32_t dsSize2 = getDataSectionSizeInBytes(src.structure);*/
	uint32_t dsSize1 = dst.dataSectionSize;
	uint32_t dsSize2 = src.dataSectionSize;
	
	CUPNP_REQUIRE(dsSize1 >= dsSize2) { dsSize2 = dsSize1; }
	
	CUPNP_REQUIRE(dst.data.isValid(dsSize1)) { return; }
	CUPNP_REQUIRE(src.data.isValid(dsSize2)) { return; }
	
	if(dsSize1 > dsSize2)
		memset(dst.data.ptr, 0, dsSize1);
	
	memcpy(dst.data.ptr, src.data.ptr, dsSize2);
}

template<typename T>
CUPNP_FUNCTION void swapData(T t1, T t2) {
	/*uint16_t dsWords1 = getDataSectionSizeInWords(t1.structure);
	uint16_t dsWords2 = getDataSectionSizeInWords(t2.structure);
	
	CUPNP_REQUIRE(dsWords1 == dsWords2);
	
	// uint16_t nPtr1 = getPointerSectionSize(t1.structure);
	// uint16_t nPtr2 = getPointerSectionSize(t2.structure);
	
	// CUPNP_REQUIRE(nPtr1 == nPtr2);
	
	uint16_t nWords = std::min(dsWords1, dsWords2);
	// uint16_t nPtrs  = std::min(nPtr1, nPtr2);*/
	
	char* data1 = (char*) t1.data.ptr;
	char* data2 = (char*) t2.data.ptr;
	
	/*for(uint16_t i = 0; i < nWords; ++i) {
		uint64_t tmp = data1[i];
		data1[i] = data2[i];
		data2[i] = tmp;
	}*/
	
	uint32_t dsBytes1 = t1.dataSectionSize;
	uint32_t dsBytes2 = t2.dataSectionSize;
	
	uint32_t nBytes = std::min(dsBytes1, dsBytes2);
	
	for(uint32_t i = 0; i < nBytes; ++i) {
		char tmp = data1[i];
		data1[i] = data2[i];
		data2[i] = tmp;
	}
	
	/*for(uint16_t i = 0; i < nPtrs; ++i)
		std::swap(*(data1 + dsWords1 + i), *(data2 + dsWords2 + i));*/
}

template<typename T>
CUPNP_FUNCTION void copyData(T t1, T t2) {
	/*uint16_t dsWords1 = getDataSectionSizeInWords(t1.structure);
	uint16_t dsWords2 = getDataSectionSizeInWords(t2.structure);
	
	CUPNP_REQUIRE(dsWords1 == dsWords2);
	
	// uint16_t nPtr1 = getPointerSectionSize(t1.structure);
	// uint16_t nPtr2 = getPointerSectionSize(t2.structure);
	
	// CUPNP_REQUIRE(nPtr1 == nPtr2);
	
	uint16_t nWords = std::min(dsWords1, dsWords2);
	// uint16_t nPtrs  = std::min(nPtr1, nPtr2);*/
	
	char* data1 = (char*) t1.data.ptr;
	char* data2 = (char*) t2.data.ptr;
	
	/*for(uint16_t i = 0; i < nWords; ++i) {
		uint64_t tmp = data1[i];
		data1[i] = data2[i];
		data2[i] = tmp;
	}*/
	
	uint32_t dsBytes1 = t1.dataSectionSize;
	uint32_t dsBytes2 = t2.dataSectionSize;
	
	uint32_t nBytes = std::min(dsBytes1, dsBytes2);
	
	for(uint32_t i = 0; i < nBytes; ++i) {
		data2[i] = data1[i];
	}
	
	/*for(uint16_t i = 0; i < nPtrs; ++i)
		std::swap(*(data1 + dsWords1 + i), *(data2 + dsWords2 + i));*/
}

/**
 * Reads the in-memory information at the given location and tries
 * to decode it as a near- or far-pointer. Returns a reference
 * to the data of the templated type.
 *
 * Warning: Assumes base to be already validated.
 */
template<typename T>
CUPNP_FUNCTION T getPointer(Location base) {		
	uint64_t ptrTagVal = ptrTag(base);
	
	Location dataLoc;
	Location structureLoc;
	
	// We only know about tag values 0, 1, 2, 3
	// Tag value 0 is structs, which we can decode
	// Tag value 1 is lists
	// Tag value 2 is far pointers
	// Tag value 3 is "other" pointers, which are currently capability pointers
	CUPNP_REQUIRE(ptrTagVal <= 3);
	
	// Capability pointers dont have a "real" data location attached
	// However, to prevent nasty edge cases, we want a valid data
	// location associated. We therefore just use the location of
	// the pointer itself.
	if(ptrTagVal == 3)
		return T(base.read<uint64_t>(), base);
			
	if(ptrTagVal == 2) {
		// Far pointer
		uint8_t decodeResult = decodeFarPtr(base, structureLoc);
	
		if(decodeResult == 0) {
			// Pointer is a far ("inter-segment") pointer, but no special landing pad
			CUPNP_REQUIRE(structureLoc.isValid(sizeof(capnp::word))) { return T(0, nullptr); }
			dataLoc = decodeNearPtr(structureLoc);
		} else if(decodeResult == 1) {
			// Landing pad is indirect far pointer to data (without special landing pad)
			// Structure information is located one word behind it		
			CUPNP_REQUIRE(structureLoc.isValid(2 * sizeof(capnp::word))) { return T(0, nullptr); }
			auto secondDecodeResult = decodeFarPtr(structureLoc, dataLoc);
			CUPNP_REQUIRE(secondDecodeResult == 0) { return T(0, nullptr); }
			
			structureLoc = structureLoc + sizeof(capnp::word);
		}
	} else {
		// Location is just a boring near pointer :)
		structureLoc = base;
		dataLoc = decodeNearPtr(structureLoc);
	}
	
	return T(structureLoc.read<uint64_t>(), dataLoc);
}

inline Location messageRoot(SegmentTable::Entry firstSegment, SegmentTable segmentRefs) {
	Location root;
	root.segmentId = 0;
	root.ptr = reinterpret_cast<unsigned char*>(firstSegment.begin());
	root.segments = segmentRefs;
	
	return root;
}

// Ensures that the location presented can hold enough data to support
// the data and pointer section size specified in "structure".
CUPNP_FUNCTION inline void validateStructPointer(uint64_t structure, Location data) {
	uint8_t tag = structure & 3u;
	CUPNP_REQUIRE(tag == 0);
	
	uint16_t dataSectionSizeInWords = structure >> 32;
	uint16_t pointerSectionSize = structure >> 48;
	
	CUPNP_REQUIRE(data.isValid((dataSectionSizeInWords + pointerSectionSize) * sizeof(capnp::word)));
}

// Ensures that the location presented can hold enough data to support
// the data and pointer section size specified in "structure".
CUPNP_FUNCTION inline void validateStructPointer(uint32_t dataSectionSize, uint16_t pointerSectionSize, Location data) {
	CUPNP_REQUIRE(data.isValid(dataSectionSize + pointerSectionSize * sizeof(capnp::word)));
}

CUPNP_FUNCTION inline void validateInterfacePointer(uint64_t structure, Location data) {
	uint8_t tag = structure & 3u;
	CUPNP_REQUIRE(tag == 3);
}

struct AnyData {
	inline AnyData(uint64_t structure, cupnp::Location data) {}
};
	
namespace internal {
	/**
	 * Helper class that performs type deduction and constructor delegation
	 * for the creation of list entries.
	 */
	template<typename T, Kind kind = KindFor<T>>
	struct AccessHelper { static_assert(sizeof(T) == 0, "Unimplemented"); };
}

template<typename T, Kind CPKind = KindFor<T>>
struct List {
	static constexpr Kind kind = Kind::LIST;
	
public:
	struct Common {
		uint32_t listSize;
		uint32_t elementSize;
		
		capnp::ElementSize sizeEnum;
		
		// uint64_t contentTag;
		uint32_t dataSectionSize;
		uint16_t pointerSectionSize;
		
		// uint64_t structure;
		cupnp::Location listStart;
		
		constexpr inline static uint64_t SINGLE_POINTER_TAG = ((uint64_t) 1) >> 48;
		constexpr inline static uint64_t SINGLE_BYTE_TAG = ((uint64_t) 1) >> 32;
		
		CUPNP_FUNCTION Common(uint64_t structure, cupnp::Location data)
		{
			uint8_t ptrTag = structure & 3u;
			CUPNP_REQUIRE(ptrTag == 1 || structure == 0);
			
			sizeEnum = static_cast<capnp::ElementSize>((structure >> 32) & 7u);
			listSize = (structure >> 35);
			
			if(sizeEnum == capnp::ElementSize::INLINE_COMPOSITE) {
				CUPNP_REQUIRE(data.isValid(sizeof(capnp::word)));
				uint64_t contentTag = data.read<uint64_t>();
				
				dataSectionSize = getDataSectionSizeInBytes(contentTag);
				pointerSectionSize = getPointerSectionSize(contentTag);
				
				elementSize = computeContentSize(contentTag, data + sizeof(capnp::word));
				listSize /= elementSize;
				elementSize *= sizeof(capnp::word);
				listStart = data + sizeof(capnp::word);
			} else {
				// contentTag = makeContentTag(sizeEnum);
				listStart = data;
				elementSize = getElementSize(sizeEnum);
				
				if(sizeEnum == capnp::ElementSize::POINTER) {
					dataSectionSize = 0;
					pointerSectionSize = 1;
				} else {
					dataSectionSize = elementSize;
					pointerSectionSize = 0;
				}				
			}
			
			CUPNP_REQUIRE(listStart.isValid(listSize * elementSize));
		}
			
		CUPNP_FUNCTION uint8_t getElementSize(capnp::ElementSize listEnum) const {
			using capnp::ElementSize;
			
			switch(listEnum) {
				case ElementSize::VOID: return 0;
				case ElementSize::BIT: return 1; // Bools
				case ElementSize::BYTE: return 1;
				case ElementSize::TWO_BYTES: return 2;
				case ElementSize::FOUR_BYTES: return 4;
				case ElementSize::EIGHT_BYTES: return 8;
				case ElementSize::POINTER: return 8;
				
				default: CUPNP_FAIL_REQUIRE("Inline composite is not associated with an element size");
			}
			
			return 0;
		}
		
		CUPNP_FUNCTION Common(const Common& other, uint32_t begin, uint32_t end) {
			CUPNP_REQUIRE(begin < other.listSize);
			CUPNP_REQUIRE(end <= other.listSize);
			
			sizeEnum = other.sizeEnum;
			listSize = end - begin;
			dataSectionSize = other.dataSectionSize;
			pointerSectionSize = other.pointerSectionSize;
			elementSize = other.elementSize;
			
			listStart = other.listStart + elementSize * begin;
		}
		
		CUPNP_FUNCTION uint32_t size() const {
			return listSize;
		}
	};
	
	template<typename ListType, typename ElementType>
	struct ListIterator {
		using difference_type = unsigned int;
		using value_type      = ElementType;
		using reference       = ElementType;
		using pointer         = ElementType*;
		using iterator_category = std::random_access_iterator_tag;
		
		const ListType& list;
		unsigned int i;
		
		CUPNP_FUNCTION ListIterator(const ListType& list, unsigned int i) : list(list), i(i) {}
		
		CUPNP_FUNCTION ListIterator& operator++() { ++i; return *this; }
		CUPNP_FUNCTION ListIterator operator++(int) { auto other = *this; ++this; return other; }
		CUPNP_FUNCTION ListIterator& operator--() { --i; return *this; }
		CUPNP_FUNCTION ListIterator operator--(int) { auto other = *this; --this; return other; }
		
		CUPNP_FUNCTION ListIterator& operator+=(unsigned int i2) { i += i2; return *this; }
		CUPNP_FUNCTION ListIterator& operator-=(unsigned int i2) { i -= i2; return *this; }
		CUPNP_FUNCTION ListIterator operator+(unsigned int i2) { return ListIterator(list, i + i2); }
		CUPNP_FUNCTION ListIterator operator-(unsigned int i2) { return ListIterator(list, i - i2); }
		
		CUPNP_FUNCTION unsigned int operator-(const ListIterator& other) { return i - other.i; }
		
		CUPNP_FUNCTION bool operator==(const ListIterator& other) { return i == other.i; }
		CUPNP_FUNCTION bool operator!=(const ListIterator& other) { return i != other.i; }
		
		CUPNP_FUNCTION ElementType operator*() { return list[i]; }
	};
	
	using Helper = ::cupnp::internal::AccessHelper<T, CPKind>;
	
	struct Reader : public Common {
		using Common::Common;
		
		using Iterator = ListIterator<Reader, typename Helper::ReaderType>;
	
		CUPNP_FUNCTION Iterator begin() { return Iterator(*this, 0); }
		CUPNP_FUNCTION Iterator end() { return iterator(*this, this -> size()); }
		
		CUPNP_FUNCTION typename Helper::ReaderType operator[] (unsigned int i) const {
			CUPNP_REQUIRE(i < this -> size());
			return Helper::get(this, i);
		}
		
		CUPNP_FUNCTION Reader slice(size_t begin, size_t end) const {
			return Reader(*this, begin, end);
		}
		
		CUPNP_FUNCTION const unsigned char* data() const {
			return this -> listStart.ptr;
		}
	};
	
	struct Builder : public Common {
		using Common::Common;
		
		using Iterator = ListIterator<Builder, typename Helper::BuilderType>;
	
		CUPNP_FUNCTION Iterator begin() { return Iterator(*this, 0); }
		CUPNP_FUNCTION Iterator end() { return iterator(*this, this -> size()); }
		
		CUPNP_FUNCTION typename Helper::BuilderType operator[] (unsigned int i) const {
			CUPNP_REQUIRE(i < this -> size());
			return Helper::get(this, i);
		}
		
		template<typename T2>
		CUPNP_FUNCTION void set(unsigned int i, T2 newVal) {
			CUPNP_REQUIRE(i < this -> size());
			return Helper::set(this, i, newVal);
		}
		
		CUPNP_FUNCTION List<T> slice(size_t begin, size_t end) {
			return Builder(*this, begin, end);
		}
		
		CUPNP_FUNCTION unsigned char* data() {
			return this -> listStart.ptr;
		}
	};
};

namespace internal {
	// Structs are stored in-line in the list, and use a structure tag placed
	// at the beginning (for in-line composite lists) or derived from the element
	// size in the list pointer.
	template<typename T>
	struct AccessHelper<T, Kind::STRUCT> {
		using ReaderType = typename T::Reader;
		using BuilderType = typename T::Builder;
		
		static CUPNP_FUNCTION ReaderType get(const typename List<T>::Reader* list, uint32_t element) {
			return ReaderType(list->dataSectionSize, list->pointerSectionSize, list->listStart + list->elementSize * element);
		}
		
		static CUPNP_FUNCTION BuilderType get(const typename List<T>::Builder* list, uint32_t element) {
			return BuilderType(list->dataSectionSize, list->pointerSectionSize, list->listStart + list->elementSize * element);
		}
		
		template<typename T2>
		static CUPNP_FUNCTION void set(const typename List<T>::Builder* list, uint32_t element, T2 value) {
			static_assert(sizeof(T) == 0, "CupnpVal<List<T>>::set is only supported for primitive T");
		}
		
		// Primitive lists may be interpreted as struct lists, with the special exception of boolean lists
		// (which have tag 1)
		// TODO: This currently creates incorrect data section sizes, so we have to disable cross-encoding of data-only structs
		static CUPNP_FUNCTION bool validList(const typename List<T>::Common* list) { return list->sizeEnum == 7 || list->sizeEnum == 6; /*return list->sizeEnum != 1;*/ }
		static CUPNP_FUNCTION ReaderType getDefault() { return ReaderType(0, nullptr); }
	};
	
	// List types are stored as pointers.
	template<typename T>
	struct AccessHelper<T, Kind::LIST> {
		using ReaderType = typename T::Reader;
		using BuilderType = typename T::Builder;
		
		static CUPNP_FUNCTION ReaderType get(const typename List<T>::Reader* list, uint32_t element) {
			return getPointer<ReaderType>(list->listStart + list->elementSize * element);
		}
		
		static CUPNP_FUNCTION BuilderType get(const typename List<T>::Builder* list, uint32_t element) {
			return getPointer<BuilderType>(list->listStart + list->elementSize * element);
		}
		
		template<typename T2>
		static CUPNP_FUNCTION void set(const typename List<T>::Builder* list, uint32_t element, T2 value) {
			static_assert(sizeof(T) == 0, "cupnp::List<T>::set is only supported for primitive T");
		}
		
		static CUPNP_FUNCTION bool validList(const typename List<T>::Common* list) { return list->sizeEnum == 7 || list->sizeEnum == 6; }
		static CUPNP_FUNCTION T getDefault() { return T(0, nullptr); }
	};
	
	// Blobs are a special case of lists.
	template<typename T>
	struct AccessHelper<T, Kind::BLOB> : public AccessHelper<T, Kind::LIST> {
	};
	
	template<typename T>
	struct AccessHelper<T, Kind::ANY_POINTER> {
		using ReaderType = typename T::Reader;
		using BuilderType = typename T::Builder;
		
		static CUPNP_FUNCTION ReaderType get(const typename List<T>::Reader* list, uint32_t element) {
			return getPointer<ReaderType>(list->listStart + list->elementSize * element);
		}
		
		static CUPNP_FUNCTION BuilderType get(const typename List<T>::Builder* list, uint32_t element) {
			return getPointer<BuilderType>(list->listStart + list->elementSize * element);
		}
		
		template<typename T2>
		static CUPNP_FUNCTION void set(const typename List<T>::Builder* list, uint32_t element, T2 value) {
			static_assert(sizeof(T) == 0, "cupnp::List<T>::set is only supported for primitive T");
		}
		
		static CUPNP_FUNCTION bool validList(const typename List<T>::Common* list) { return list->sizeEnum == 7 || list->sizeEnum == 6; }
		static CUPNP_FUNCTION T getDefault() { return T(0, nullptr); }
	};
	
	template<typename T>
	struct AccessHelper<T, Kind::INTERFACE> {
		using ReaderType = T;
		using BuilderType = T;
		
		static CUPNP_FUNCTION ReaderType get(const typename List<T>::Reader* list, uint32_t element) {
			return getPointer<ReaderType>(list->listStart + list->elementSize * element);
		}
		
		static CUPNP_FUNCTION BuilderType get(const typename List<T>::Builder* list, uint32_t element) {
			return getPointer<BuilderType>(list->listStart + list->elementSize * element);
		}
		
		template<typename T2>
		static CUPNP_FUNCTION void set(const typename List<T>::Builder* list, uint32_t element, T2 value) {
			static_assert(sizeof(T) == 0, "cupnp::List<T>::set is only supported for primitive T");
		}
		
		static CUPNP_FUNCTION bool validList(const typename List<T>::Common* list) { return list->sizeEnum == 7 || list->sizeEnum == 6; }
		static CUPNP_FUNCTION T getDefault() { return T(0, nullptr); }
	};
	
	// Primitives, like structs, are stored in-line in the list, and must be accessed
	// directly. Contrary to all other value types, we support setting these as well.
	template<typename T>
	struct AccessHelper<T, Kind::PRIMITIVE> {
		using ReaderType = T;
		using BuilderType = T;
		
		static CUPNP_FUNCTION T get(const typename List<T>::Common* list, uint32_t element) {
			return (list->listStart + list->elementSize * element).template read<T>();
		}
		
		static CUPNP_FUNCTION void set(const typename List<T>::Builder* list, uint32_t element, T value) {
			(list->listStart + list->elementSize * element).template write<T>(value);
		}
		
		// Primitive lists can be inline composite lists (tag 7) or any non-pointer list (tag 6)
		// For inline composite lists, the element size 
		static CUPNP_FUNCTION bool validList(const typename List<T>::Common* list ) {
			// Boolean and non-boolean interpretations are incompatible
			if(list->sizeEnum == 1)
				return false;
			
			// Pointer lists are not allowed as primitivers
			if(list->sizeEnum == 6)
				return false;
			
			// Inline composite lists must at least hold the contents
			if(list->sizeEnum == 7)
				return list->elementSize >= sizeof(T);
			
			// Non-composite list must match exactly
			return list->elementSize == sizeof(T);
		}
		static CUPNP_FUNCTION T getDefault() { return (T) 0; }
	};
	
	// Primitives, like structs, are stored in-line in the list, and must be accessed
	// directly. Contrary to all other value types, we support setting these as well.
	template<typename T>
	struct AccessHelper<T, Kind::ENUM> {
		using ReaderType = T;
		using BuilderType = T;
		
		static CUPNP_FUNCTION T get(const typename List<T>::Common* list, uint32_t element) {
			return static_cast<T>((list->listStart + list->elementSize * element).template read<uint16_t>());
		}
		
		static CUPNP_FUNCTION void set(const typename List<T>::Builder* list, uint32_t element, T value) {
			(list->listStart + list->elementSize * element).template write<uint16_t>(static_cast<uint16_t>(value));
		}
		
		// Primitive lists can be inline composite lists (tag 7) or any non-pointer list (tag 6)
		// For inline composite lists, the element size 
		static CUPNP_FUNCTION bool validList(const typename List<T>::Common* list ) {
			// Boolean and non-boolean interpretations are incompatible
			if(list->sizeEnum == 1)
				return false;
			
			// Pointer lists are not allowed as primitivers
			if(list->sizeEnum == 6)
				return false;
			
			// Inline composite lists must at least hold the contents
			if(list->sizeEnum == 7)
				return list->elementSize >= sizeof(uint16_t);
			
			// Non-composite list must match exactly
			return list->elementSize == sizeof(uint16_t);
		}
		static CUPNP_FUNCTION T getDefault() { return (T) 0; }
	};
	
	// Bool values need some special handling, as they are not aligned on byte boundaries.
	template<>
	struct AccessHelper<bool, Kind::PRIMITIVE> {
		using ReaderType = bool;
		using BuilderType = bool;
		
		static CUPNP_FUNCTION bool get(const List<bool>::Common* list, uint32_t element) {
			auto loc = list->listStart + element / 8;
			uint8_t byteVal = loc.read<uint8_t>();
			
			return (byteVal >> (element % 8)) & 1;
		}
		
		static CUPNP_FUNCTION void set(List<bool>::Builder* list, uint32_t element, bool val) {
			auto loc = list->listStart + element / 8;
			uint8_t byteVal = loc.read<uint8_t>();
			
			uint8_t mask = 1 << (element % 8);
			if(val)
				byteVal |= mask;
			else
				byteVal &= !mask;
			
			loc.write(byteVal);
		}
		
		static CUPNP_FUNCTION bool validList(List<bool>::Common* list) { return list->sizeEnum == capnp::ElementSize::BIT; }
		static CUPNP_FUNCTION bool getDefault() { return false; }
	};
}
	
struct Data {
	static constexpr Kind kind = Kind::BLOB;
	
	struct Reader  {
		List<uint8_t>::Reader backend;
	
		CUPNP_FUNCTION Reader(uint64_t structure, cupnp::Location data) : backend(structure, data) {
			CUPNP_REQUIRE(backend.dataSectionSize == 1 && backend.pointerSectionSize == 0);
		}
	
		CUPNP_FUNCTION const unsigned char* data() const {
			return backend.listStart.ptr;
		}
	
		CUPNP_FUNCTION uint32_t size() { return backend.size(); }
	};
	
	struct Builder {
		List<uint8_t>::Builder backend;
	
		CUPNP_FUNCTION Builder(uint64_t structure, cupnp::Location data) : backend(structure, data) {
			CUPNP_REQUIRE(backend.dataSectionSize == 1 && backend.pointerSectionSize == 0);
		}
	
		CUPNP_FUNCTION unsigned char* data() const {
			return backend.listStart.ptr;
		}
	
		CUPNP_FUNCTION uint32_t size() { return backend.size(); }
	};
};

struct Text {
	static constexpr Kind kind = Kind::BLOB;
	
	struct Reader  {
		List<uint8_t>::Reader backend;
	
		CUPNP_FUNCTION Reader(uint64_t structure, cupnp::Location data) : backend(structure, data) {
			CUPNP_REQUIRE(backend.dataSectionSize == 1 && backend.pointerSectionSize == 0);
		}
	
		CUPNP_FUNCTION const unsigned char* data() const {
			return backend.listStart.ptr;
		}
	
		CUPNP_FUNCTION uint32_t size() {
			CUPNP_REQUIRE(backend.size() >= 1);
			return backend.size() - 1;
		}
	};
	
	struct Builder {
		List<uint8_t>::Builder backend;
	
		CUPNP_FUNCTION Builder(uint64_t structure, cupnp::Location data) : backend(structure, data) {
			CUPNP_REQUIRE(backend.dataSectionSize == 1 && backend.pointerSectionSize == 0);
		}
	
		CUPNP_FUNCTION unsigned char* data() const {
			return backend.listStart.ptr;
		}
	
		CUPNP_FUNCTION uint32_t size() {
			CUPNP_REQUIRE(backend.size() >= 1);
			return backend.size() - 1;
		}
	};
};

struct AnyPointer {
	static constexpr Kind kind = Kind::ANY_POINTER;
	
	struct Reader {
		uint64_t structure;
		cupnp::Location data;
		
		CUPNP_FUNCTION Reader(uint64_t structure, cupnp::Location data) :
			structure(structure), data(data)
		{}
	};
	
	struct Builder {
		uint64_t structure;
		cupnp::Location data;
		
		CUPNP_FUNCTION Builder(uint64_t structure, cupnp::Location data) :
			structure(structure), data(data)
		{}
	};
};


}