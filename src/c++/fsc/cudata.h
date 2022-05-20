#pragma once

#include <cupnp/cupnp.h>
#include "kernels.h"
#include "data.h"

namespace fsc {
	
namespace internal {
	
/**
 * Takes a table to potentially const memory and changes it to non-const pointers.
 * This is just for internal storage, constness of the memory should still be assumed
 * unless known to be mutable.
 */
template<typename T>
kj::Array<kj::ArrayPtr<capnp::word>> coerceSegmentTableToNonConst(T table) {	
	auto outputTable = heapArray<kj::ArrayPtr<capnp::word>>(table.size());
	
	for(size_t i = 0; i < table.size(); ++i) {
		auto maybeConstPtr = table[i].begin();
		capnp::word* nonConstPtr = const_cast<capnp::word*>(maybeConstPtr);
		size_t nWords = table[i].size();
		
		outputTable[i] = kj::ArrayPtr<capnp::word>(nonConstPtr, nWords);
	}
	
	return outputTable;
}

}

template<typename T>
CupnpMessage {
	kj::Array<kj::ArrayPtr<capnp::word>> segmentTable;
	
	T root() {
		return cupnp::messageRoot<T>(segmentTable[0], segmentTable);
	}
	
	CupnpMessage(capnp::MessageBuilder& builder) :
		segmentTable(coerceSegmentTableToNonConst(builder.getSegmentsForOutput()))
	{}
	
	CupnpMessage(capnp::MessageReader& reader)
	{
		KJ_REQUIRE(std::is_const<T>::value, "Can only build const messages from message readers");
		
		kj::Vector<kj::ArrayPtr<const word>> segments;
		
		size_t segmentId = 0;
		while(true) {
			KJ_IF_MAYBE(pSegment, reader.getSegment(segmentId++)) {
				segment.pushBack(*pSegment);
			} else {
				break;
			}
		}
		
		segmentTable = coerceSegmentTableToNonConst(segments.releaseAsArray());
	}
	
	CupnpMessage(kj::ArrayPtr<const kj::ArrayPtr<const word>> segments) :
		segmentTable(coerceSegmentTableToNonConst(segments))
	{}
};

template<typename CuT = cupnp::CuFor<T>, typename T>
CupnpMessage<CuT> cupnpMessageFromTemporary(Temporary<T>& tmp) {
	return CupnpMessage<CuT>(*(tmp.holder));
}
	
template<typename T, typename Device>
struct MapToDevice<CupnpMessage<T>, Device> {
	using Msg = CupnpMessage<T>;
	
	Msg& original;
	Device& device;
	
	kj::Array<MappedData<capnp::word>> deviceSegments;
	
	kj::Array<kj::ArrayPtr<capnp::word>> hostSegmentTable;
	MappedData<kj::ArrayPtr<capnp::word>> deviceSegmentTable;
	
	MapToDevice(Msg& original, Device& device) :
		original(original), device(device)
	{
		size_t nSegments = original.segmentTable.size();
		
		// Allocate segments
		{
			auto builder = kj::heapArrayBuilder<MappedData<capnp::word>>(nSegments);
			
			for(size_t i = 0; i < nSegments; ++i) {
				kj::ArrayPtr<capnp::word> segment = original.segmentTable[i];
				
				builder.add(device, segment.begin(), segment.size());
			}
			
			deviceSegments = builder.releaseAsArray();
		}
		
		// Gather device pointers
		{
			auto builder = kj::heapArrayBuilder<kj::ArrayPtr<capnp::word>>(nSegments);
			
			for(size_t i = 0; i < nSegments; ++i) {
				builder.add(kj::ArrayPtr<capnp::word>(
					deviceSegments[i].devicePtr,
					deviceSegments[i].size
				));
			}
			
			hostSegmentTable = builder.releaseAsArray();
		}
		
		// Map segment table onto device
		deviceSegmentTable = MappedData(Device, hostSegmentTable.begin(), hostSegmentTable.size());		
	}
	
	void updateHost() {
		if(std::is_const<T>::value)
			return;
	
		for(auto& segment : deviceSegments) {
			segment.updateHost();
		}
	}
	
	void updateDevice() {
		for(auto& segment : deviceSegments) {
			segment.updateDevice();
		}
	}
	
	T get() {
		kj::ArrayPtr<kj::ArrayPtr<capnp::word>> ptrSegmentTable(deviceSegmentTable.devicePtr, deviceSegmentTable.size);
		kj::ArrayPtr<capnp::word> firstSegment = hostSegmentTable[0];
		
		return cupnp::messageRoot<T>(firstSegment, ptrSegmentTable);
	}
};

}