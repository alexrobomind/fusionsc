#include <capnp/common.h>
#include <capnp/list.h>

#include <kj/common.h>

#include "gpu_defs.h"

#pragma once

# ifdef CUPNP_GPUCC
	//TODO: Better error handling for device code?
	# define CUPNP_REQUIRE(...) (void)0
	# define CUPNP_FAIL_REQUIRE(...) (void)0
# else
	# include <kj/debug.h>

	# define CUPNP_REQUIRE(...) KJ_REQUIRE((__VA_ARGS__))
	# define CUPNP_FAIL_REQUIRE(...) KJ_FAIL_REQUIRE((__VA_ARGS__))
# endif

namespace cupnp {
	// Value struct
	template<typename T>
	struct CupnpVal { static_assert(sizeof(T) == 0, "Unimplemented"); };
			
	
	/**
	 * Helper class that performs type deduction and constructor delegation
	 * for the creation of list entries.
	 */
	template<typename T, capnp::Kind Kind = CAPNP_KIND(T)>
	struct ListHelper { static_assert(sizeof(T) == 0, "Unimplemented"); };
	
	// struct Message {
	//	/*unsigned int nSegments;
	//	capnp::word** segments; // Size is 2 * nSegments*/
	//	kj::ArrayPtr<kj::ArrayPtr<capnp::word>> segments;
	//};
	
	struct Location {
		unsigned int segmentId;
		unsigned char* ptr;
		
		kj::ArrayPtr<kj::ArrayPtr<capnp::word>> segments;
		
		template<typename T>
		T read() const {
			// Assume GPU is little-endian
			# ifdef CUPNP_GPUCC
				return *(reinterpret_cast<T*>(ptr));
			# else
				return reinterpret_cast<capnp::_::WireValue<T>*>(ptr)->get();
			# endif
		}
		
		template<typename T>
		void write(T newVal) {
			// Assume GPU is little-endian
			# ifdef CUPNP_GPUCC
				*(reinterpret_cast<T*>(ptr)) = newVal;
			# else
				reinterpret_cast<capnp::_::WireValue<T>*>(ptr)->set(newVal);
			# endif
		}
		
		inline Location operator+(int32_t shift) const {
			Location l2;
			l2.segmentId = segmentId;
			l2.ptr = ptr + shift;
			l2.segments = segments;
			
			return l2;
		}
		
		inline bool isValid(size_t size) {
			if(ptr == nullptr)
				return false;
			
			if(segments == nullptr)
				return true;
			
			if(segmentId >= segments->nSegments)
				return false;
			
			auto start = reinterpret_cast<unsigned char*>(/*segments->segments[2 * segmentId]*/segments[segmentId].begin());
			auto end   = reinterpret_cast<unsigned char*>(/*segments->segments[2 * segmentId + 1]*/segments[segmentId].end());
			
			if(ptr < start)
				return false;
			
			return ptr + size <= end;
		}
	};
	
	template<typename T>
	inline CupnpVal<T> messageRoot(kj::ArrayPtr<kj::ArrayPtr<capnp::word>> segments) {
		Location root;
		root.segmentId = 0;
		root.ptr = reinterpret_cast<unsigned char*>(segments[0].begin());
		root.segments = segments;
		
		return getPointer<T>(root);
	}
	
	inline kj::Array<size_t> calculateSizes(kj::ArrayPtr<const kj::ArrayPtr<const capnp::word>> segments) {
		auto sizes = kj::heapArrayBuilder<size_t>(segments.size());
		for(auto segment : segments)
			sizes.add(segment.size());
		
		return sizes.finish();
	}
	
	inline auto deviceMemcpy(void* dst, const void* src, size_t nBytes) {
		# ifdef CUPNP_WITH_HIP
			return hipMemcpy(dst.begin(), src.begin(), nBytes, cudaMemcpyDefault);
		# else
			# ifdef CUPNP_WITH_CUDA
				return cudaMemcpy(dst.begin(), src.begin(), nBytes, hipMemcpyDefault);
			# else
				memcpy(dst.begin(), src.begin(), nBytes);
				return (int) 0;
			# endif
		# endif
	}
	
	inline auto deviceMemcpy(kj::ArrayPtr<capnp::word> dst, const kj::ArrayPtr<const capnp::word> src) {
		CUPNP_REQUIRE(dst.size() >= src.size());
		const auto nBytes = src.size() * sizeof(capnp::word);
		
		return deviceMemcpy(dst.begin(), src.begin(), nBytes);
	}
	
	inline auto deviceMemcpy(kj::ArrayPtr<capnp::word> dst, const kj::ArrayPtr<capnp::word> src) {
		return deviceMemcpy(dst, src.asConst());
	}
	
	template<typename T1, typename T2>
	inline auto deviceMemcpyAll(T1 dst, const T2 src) {
		CUPNP_REQUIRE(dst.size() == src.size()); 
		
		for(size_t i = 0; i < dst.size(); ++i) { 
			auto err = deviceMemcpy(dst[i], src[i]); 
			
			if(err != 0) 
				return err; 
		} 
	}
	
	struct Message {
		// Host-located array of segments (which can individually be device-located)
		kj::Array<kj::Array<capnp::word>> segments;
		
		// Device-located array of segments
		kj::Array<kj::ArrayPtr<capnp::word>> segmentRefs;
		
		template<typename Allocator>
		Message(kj::ArrayPtr<size_t> sizes, bool onDevice) {
			auto segmentsBuilder = kj::heapArrayBuilder<kj::Array<capnp::word>>(sizes.size());
			for(size_t size : sizes) {
				if(onDevice)
					segmentsBuilder.add(deviceArray<capnp::word>(size));
				else
					segmentsBuilder.add(kj::heapArray<capnp::word>(size));
			}
			segments = segmentsBuilder.finish();
			
			auto hostSegmentRefs = kj::heapArray<kj::ArrayPtr<capnp::word>>(sizes.size());			
			for(size_t i = 0; i < sizes.size(); ++i) {
				hostSegmentRefs[i] = segments[i].asPtr();
			}
			
			segmentRefs = onDevice ?
				deviceArray  <kj::ArrayPtr<capnp::word>>(sizes.size()) :
				kj::heapArray<kj::ArrayPtr<capnp::word>>(sizes.size())
			;
			deviceMemcpy(segmentRefs, hostSegmentRefs);
		}
			
		template<typename Allocator>
		Message(kj::ArrayPtr<const kj::ArrayPtr<const capnp::word>> hostSegments, Allocator allocator) :
			Message(calculateSizes(hostSegments), allocator)
		{			
			deviceMemcpyAll(segments, hostSegments);
		}
		
		kj::Array<size_t> sizes() {
			return computeSizes(segments);
		}
	};
	
	inline auto deviceMemcpy(Message dst, const Message src) {
		return deviceMemcpyAll(dst.segments, src.segments);
	}
	
	/**
	 * Returns the pointer tag, which is stored in its
	 * 2 least significant bits.
	 */
	inline unsigned char ptrTag(Location in) {
		uint64_t nativeVal = in.read<uint64_t>();
		
		return nativeVal & 3u;
	}

	/**
	 * Given a location pointing to an intra-segment pointer,
	 * returns the location pointed at by the pointer.
	 */
	inline Location decodeNearPtr(Location in) {
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
	inline unsigned char decodeFarPtr(const Location& in, Location& out) {
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
		CUPNP_REQUIRE(message != nullptr);
		
		// Far pointer decoding
		uint8_t landingPadType;
		uint32_t landingPadOffset;
		uint32_t segmentId;
		
		landingPadType = (nativeVal >> 2) & 1; // B
		landingPadOffset = (nativeVal & ((1ull << 32) - 1)) >> 3; // C
		segmentId = nativeVal >> 32; // D
		
		CUPNP_REQUIRE(segmentId < message->nSegments);
		out.ptr = reinterpret_cast<unsigned char*>(message->segments[2 * segmentId] + landingPadOffset); // Offset calculation is in 8-byte words
		CUPNP_REQUIRE(out.ptr < reinterpret_cast<unsigned char*>(message->segments[2 * segmentId + 1]));
		
		out.segmentId = segmentId;
		out.segments = message;
		
		return landingPadType;
	}

	/**
	 * Creates a value holder of the specified type from a single-segment capnproto message.
	 * The memory contents pointed to by data must be a single near-pointer (can be list or structure pointer),
	 * which points to the contents of the default value data.
	 */
	template<typename T>
	const CupnpVal<T> getDefaultValue(const capnp::word* data) {
		// Default value
		Location structureLoc;
		structureLoc.ptr = reinterpret_cast<unsigned char*>(const_cast<capnp::word*>(data));
		structureLoc.segments = nullptr;
		structureLoc.segmentId = -1;
		
		Location dataLoc = decodeNearPtr(structureLoc);
		
		return CupnpVal<T>(structureLoc.read<uint64_t>(), dataLoc);
	}

	/**
	 * Reads a pointer-type field of a structure in read-only mode. If the pointer is 0
	 * (or out of bounds of the structure), the default value will be returned instread.
	 */
	template<typename T, uint32_t offset>
	const CupnpVal<T> getPointerField(uint64_t structure, Location data, const capnp::word* defaultValue) {
		uint16_t dataSectionSizeInWords = structure >> 32;
		uint16_t pointerSectionSize = structure >> 48;
		
		if(offset >= pointerSectionSize) {
			return getDefaultValue<T>(defaultValue);
		}
		
		Location ptrLoc = data + sizeof(capnp::word) * (dataSectionSizeInWords + offset);

		bool isDefault = ptrLoc.read<uint64_t>() == 0;
		if(isDefault) {
			return getDefaultValue<T>(defaultValue);
		}
		
		return getPointer<T>(ptrLoc);
	}
	
	template<typename T, unsigned int size = sizeof(T)> struct EncodedType_ {};
	template<typename T> struct EncodedType_<T, 1> { using Type = uint8_t; };
	template<typename T> struct EncodedType_<T, 2> { using Type = uint16_t; };
	template<typename T> struct EncodedType_<T, 4> { using Type = uint32_t; };
	template<typename T> struct EncodedType_<T, 8> { using Type = uint64_t; };
	
	template<typename T>
	using EncodedType = typename EncodedType_<T>::Type;
	
	template<typename T>
	EncodedType<T> encodePrimitive(T value) {
		EncodedType<T> result;
		memcpy(&result, &value, sizeof(T));
		return result;
	}
	
	template<typename T>
	T decodePrimitive(EncodedType<T> value) {
		T result;
		memcpy(&result, &value, sizeof(T));
		return result;
	}
	
	/**
	 * Reads a primitive value from the wire data.
	 */
	template<typename T, uint32_t offset>
	const T getPrimitiveField(uint64_t structure, Location data, T defaultValue) {
		uint16_t dataSectionSizeInWords = structure >> 32;
		
		// If the data section can not fully hold this value, retrn the default value instead
		if(sizeof(T) * (offset + 1) > sizeof(capnp::word) * dataSectionSizeInWords) {
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
	const T getPrimitiveField(uint64_t structure, Location data) {
		uint16_t dataSectionSizeInWords = structure >> 32;
		
		// If the data section can not fully hold this value, retrn the default value instead
		if(sizeof(T) * (offset + 1) > sizeof(capnp::word) * dataSectionSizeInWords) {
			return 0;
		}
		
		// Read the bitwise representation from the wire
		EncodedType<T> wireData = (data + offset * sizeof(T)).read<EncodedType<T>>();

		// Perform a bitcast
		return decodePrimitive<T>(wireData);
	}
	
	template<typename T, uint32_t offset>
	void setPrimitiveField(uint64_t structure, Location data, T defaultValue, T value) {
		uint16_t dataSectionSizeInWords = structure >> 32;
		
		CUPNP_REQUIRE(sizeof(T) * (offset + 1) <= sizeof(capnp::word) * dataSectionSizeInWords);
		
		EncodedType<T> wireData       = encodePrimitive(value);
		
		// XOR the bitwise representation with the default value
		EncodedType<T> encodedDefault = encodePrimitive(defaultValue);
		wireData ^= encodedDefault;
		
		(data + offset * sizeof(T)).write(wireData);
	}
	
	template<typename T, uint32_t offset>
	void setPrimitiveField(uint64_t structure, Location data, T value) {
		uint16_t dataSectionSizeInWords = structure >> 32;
		
		CUPNP_REQUIRE(sizeof(T) * (offset + 1) <= sizeof(capnp::word) * dataSectionSizeInWords);
		
		EncodedType<T> wireData       = encodePrimitive(value);
		(data + offset * sizeof(T)).write(wireData);
	}
	
	template<uint32_t offset>
	const uint16_t getDiscriminant(uint64_t structure, Location data) {
		return getPrimitiveField<uint16_t, offset>(structure, data);
	}
	
	template<uint32_t offset>
	const void setDiscriminant(uint64_t structure, Location data, uint16_t newVal) {
		setPrimitiveField<uint16_t, offset>(structure, data, newVal);
	}

	/**
	 * Returns a mutable reference to the data in the given pointer field.
	 * Fails if the field is out-of-bounds or 0, as the default value storage
	 * is not mutable.
	 */
	template<typename T, uint32_t offset>
	CupnpVal<T> mutatePointerField(uint64_t structure, Location data) {
		uint16_t dataSectionSizeInWords = structure >> 32;
		uint16_t pointerSectionSize = structure >> 48;
		
		CUPNP_REQUIRE(offset < pointerSectionSize);
		
		Location ptrLoc = data + sizeof(capnp::word) * (dataSectionSizeInWords + offset);
		bool isDefault = ptrLoc.read<uint64_t>() == 0;
		
		CUPNP_REQUIRE(!isDefault);
		return getPointer<T>(ptrLoc);
	}
	
	template<uint32_t offset>
	bool hasPointerField(uint64_t structure, Location data) {
		uint16_t dataSectionSizeInWords = structure >> 32;
		uint16_t pointerSectionSize = structure >> 48;
		
		if(offset >= pointerSectionSize)
			return false;
		
		Location ptrLoc = data + sizeof(capnp::word) * (dataSectionSizeInWords + offset);
		bool isDefault = ptrLoc.read<uint64_t>() == 0;
		
		return !isDefault;
	}

	/**
	 * Compute the size (in 8-byte words) of the data region pointed to by the
	 * given data with the provided structure information.
	 *
	 * Note: Currently only accepts structure pointers.
	 */
	inline uint32_t computeContentSize(uint64_t structure, Location data) {
		uint8_t tag = structure && 3u;
		if(tag == 0) {
			uint16_t dataSectionSizeInWords = structure >> 32;
			uint16_t pointerSectionSize = structure >> 48;
			return dataSectionSizeInWords + pointerSectionSize;
		} else {
			CUPNP_FAIL_REQUIRE("Implement missing nested list size");
			
			KJ_UNREACHABLE
		}
	}

	/**
	 * Reads the in-memory information at the given location and tries
	 * to decode it as a near- or far-pointer. Returns a reference
	 * to the data of the templated type.
	 *
	 * Warning: Assumes base to be already validated.
	 */
	template<typename T>
	CupnpVal<T> getPointer(Location base) {		
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
			return CupnpVal<T>(base.read<uint64_t>(), base);
				
		if(ptrTagVal == 2) {
			// Far pointer
			uint8_t decodeResult = decodeFarPtr(base, structureLoc);
		
			if(decodeResult == 0) {
				// Pointer is a far ("inter-segment") pointer, but no special landing pad
				CUPNP_REQUIRE(structureLoc.isValid(sizeof(capnp::word)));
				dataLoc = decodeNearPtr(structureLoc);
			} else if(decodeResult == 1) {
				// Landing pad is indirect far pointer to data (without special landing pad)
				// Structure information is located one word behind it		
				CUPNP_REQUIRE(structureLoc.isValid(2 * sizeof(capnp::word)));
				auto secondDecodeResult = decodeFarPtr(structureLoc, dataLoc);
				CUPNP_REQUIRE(secondDecodeResult == 0);
				
				structureLoc = structureLoc + sizeof(capnp::word);
			}
		} else {
			// Location is just a boring near pointer :)
			structureLoc = base;
			dataLoc = decodeNearPtr(structureLoc);
		}
		
		return CupnpVal<T>(structureLoc.read<uint64_t>(), dataLoc);
	}
	
	// Ensures that the location presented can hold enough data to support
	// the data and pointer section size specified in "structure".
	inline void validateStructPointer(uint64_t structure, Location data) {
		uint8_t tag = structure & 3u;
		CUPNP_REQUIRE(tag == 0);
		
		uint16_t dataSectionSizeInWords = structure >> 32;
		uint16_t pointerSectionSize = structure >> 48;
		
		CUPNP_REQUIRE(data.isValid((dataSectionSizeInWords + pointerSectionSize) * sizeof(capnp::word)));
	}
	
	inline void validateInterfacePointer(uint64_t structure, Location data) {
		uint8_t tag = structure & 3u;
		CUPNP_REQUIRE(tag == 3);
	}

	template<typename T, capnp::Kind CPKind>
	struct CupnpVal<capnp::List<T, CPKind>> {
		uint32_t listSize;
		uint32_t elementSize;
		
		capnp::ElementSize sizeEnum;
		
		uint64_t contentTag;
		
		uint64_t structure;
		cupnp::Location listStart;
		
		constexpr inline static uint64_t SINGLE_POINTER_TAG = ((uint64_t) 1) >> 48;
		constexpr inline static uint64_t SINGLE_BYTE_TAG = ((uint64_t) 1) >> 32;
		
		CupnpVal(uint64_t structure, cupnp::Location data) :
			structure(structure)
		{
			uint8_t ptrTag = structure & 3u;
			CUPNP_REQUIRE(ptrTag == 1 || structure == 0);
			
			sizeEnum = static_cast<capnp::ElementSize>((structure >> 32) & 7u);
			listSize = (structure >> 35);
			
			if(sizeEnum == capnp::ElementSize::INLINE_COMPOSITE) {
				CUPNP_REQUIRE(data.isValid(sizeof(capnp::word)));
				contentTag = data.read<uint64_t>();
				
				elementSize = computeContentSize(contentTag, data + sizeof(capnp::word));
				listSize /= elementSize;
				elementSize *= sizeof(capnp::word);
				listStart = data + sizeof(capnp::word);
			} else {
				contentTag = makeContentTag(sizeEnum);
				listStart = data;
				elementSize = getElementSize(sizeEnum);
			}
			
			CUPNP_REQUIRE(listStart.isValid(listSize * elementSize));
		}
			
		uint8_t getElementSize(capnp::ElementSize listEnum) const {
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
		}
		
		uint64_t makeContentTag(capnp::ElementSize listEnum) const {
			CUPNP_REQUIRE(listEnum < capnp::ElementSize::POINTER);
			
			if(listEnum == capnp::ElementSize::POINTER) {
				return SINGLE_POINTER_TAG;
			}
			
			return ((uint64_t) getElementSize(listEnum)) >> 32;
		}
		
		auto operator[] (unsigned int i) {
			return cupnp::ListHelper<T, CPKind>::get(this, i);
		}
		
		const auto operator[] (unsigned int i) const {
			return cupnp::ListHelper<T, CPKind>::get(this, i);
		}
		
		template<typename T2>
		void set(unsigned int i, T2 newVal) {
			return cupnp::ListHelper<T, CPKind>::set(this, i, newVal);
		}
		
		uint32_t size() const {
			return listSize;
		}
	}; 
	
	// Structs are stored in-line in the list, and use a structure tag placed
	// at the beginning (for in-line composite lists) or derived from the element
	// size in the list pointer.
	template<typename T>
	struct ListHelper<T, capnp::Kind::STRUCT> {
		static CupnpVal<T> get(CupnpVal<capnp::List<T>>* list, uint32_t element) {
			return CupnpVal<T>(list->contentTag, list->listStart + list->elementSize * element);
		}
		
		static const CupnpVal<T> get(const CupnpVal<capnp::List<T>>* list, uint32_t element) {
			return CupnpVal<T>(list->contentTag, list->listStart + list->elementSize * element);
		}
		
		template<typename T2>
		static void set(CupnpVal<capnp::List<T>>* list, uint32_t element, T2 value) {
			static_assert(sizeof(T) == 0, "CupnpVal<List<T>>::set is only supported for primitive T");
		}
		
		// Primitive lists may be interpreted as struct lists, with the special exception of boolean lists
		// (which have tag 1)
		static bool validList(CupnpVal<capnp::List<T>>*) { return list->sizeEnum != 1; }
	};
	
	// Blob type values are stored as pointers without shared structure information.
	// These pointers are just decoded like pointer fields.
	template<typename T>
	struct ListHelper<T, capnp::Kind::BLOB> {
		static CupnpVal<T> get(CupnpVal<capnp::List<T>>* list, uint32_t element) {
			return getPointer<T>(list->listStart + list->elementSize * element);
		}
		
		static const CupnpVal<T> get(const CupnpVal<capnp::List<T>>* list, uint32_t element) {
			return getPointer<T>(list->listStart + list->elementSize * element);
		}
		
		template<typename T2>
		static T set(CupnpVal<capnp::List<T>>* list, uint32_t element, T2 value) {
			static_assert(sizeof(T) == 0, "CupnpVal<List<T>>::set is only supported for primitive T");
		}
		
		static bool validList(CupnpVal<capnp::List<T>>*) { return list->sizeEnum == 7 || list->sizeEnum == 6; }
	};
	
	// Similarly to Blob types, lists are stored as pointers as well (the term similar is misleading,
	// as the specification explicitly mentions that blob types are a special case of list encoding).
	template<typename T>
	struct ListHelper<T, capnp::Kind::LIST> {
		static CupnpVal<T> get(CupnpVal<capnp::List<T>>* list, uint32_t element) {
			return getPointer<T>(list->listStart + list->elementSize * element);
		}
		
		static const CupnpVal<T> get(const CupnpVal<capnp::List<T>>* list, uint32_t element) {
			return getPointer<T>(list->listStart + list->elementSize * element);
		}
		
		template<typename T2>
		static T set(CupnpVal<capnp::List<T>>* list, uint32_t element, T2 value) {
			static_assert(sizeof(T) == 0, "CupnpVal<List<T>>::set is only supported for primitive T");
		}
		
		static bool validList(CupnpVal<capnp::List<T>>*) { return list->sizeEnum == 7 || list->sizeEnum == 6; }
	};
	
	// Primitives, like structs, are stored in-line in the list, and must be accessed
	// directly. Contrary to all other value types, we support setting these as well.
	template<typename T>
	struct ListHelper<T, capnp::Kind::PRIMITIVE> {
		static T get(const CupnpVal<capnp::List<T>>* list, uint32_t element) {
			return (list->listStart + list->elementSize * element).read<T>();
		}
		
		static void set(CupnpVal<capnp::List<T>>* list, uint32_t element, T value) {
			(list->listStart + list->elementSize * element).write<T>(value);
		}
		
		// Primitive lists can be inline composite lists (tag 7) or any non-pointer list (tag 6)
		// For inline composite lists, the element size 
		static bool validList(CupnpVal<capnp::List<T>>*) {
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
	};
	
	// Bool values need some special handling, as they are not aligned on byte boundaries.
	template<>
	struct ListHelper<bool, capnp::Kind::PRIMITIVE> {
		static bool get(const CupnpVal<capnp::List<bool>>* list, uint32_t element) {
			auto loc = list->listStart + element / 8;
			uint8_t byteVal = loc.read<uint8_t>();
			
			return (byteVal >> (element % 8)) & 1;
		}
		
		static bool set(CupnpVal<capnp::List<bool>>* list, uint32_t element, bool val) {
			auto loc = list->listStart + element / 8;
			uint8_t byteVal = loc.read<uint8_t>();
			
			uint8_t mask = 1 << (element % 8);
			if(val)
				byteVal |= mask;
			else
				byteVal &= !mask;
			
			loc.write(byteVal);
		}
		
		static bool validList(CupnpVal<capnp::List<bool>>* list) { return list->sizeEnum == capnp::ElementSize::BIT; }
	};
	
	template<>
	struct CupnpVal<capnp::Data> {
		CupnpVal<capnp::List<uint8_t>> backend;
		
		CupnpVal(uint64_t structure, cupnp::Location data) : backend(structure, data) {
			CUPNP_REQUIRE(backend.contentTag == CupnpVal<capnp::List<uint8_t>>::SINGLE_BYTE_TAG);
		}

		unsigned char* data() {
			return backend.listStart.ptr;
		}
		
		const unsigned char* data() const {
			return backend.listStart.ptr;
		}
		
		uint32_t size() { return backend.size(); }
	};

	template<>
	struct CupnpVal<capnp::Text> {
		CupnpVal<capnp::List<uint8_t>> backend;
		
		CupnpVal(uint64_t structure, cupnp::Location data) : backend(structure, data) {
			CUPNP_REQUIRE(backend.contentTag == CupnpVal<capnp::List<uint8_t>>::SINGLE_BYTE_TAG);
		}

		unsigned char* data() {
			return backend.listStart.ptr;
		}
		
		const unsigned char* data() const {
			return backend.listStart.ptr;
		}
		
		uint32_t size() { return backend.size(); }
	};
}
