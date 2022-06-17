#pragma once

#include <cupnp/cupnp.h>
#include "kernels.h"
#include "data.h"

#include <capnp/serialize.h>

namespace fsc {
	
namespace internal {
	
/**
 * Takes a table to potentially const memory and changes it to non-const pointers.
 * This is just for internal storage, constness of the memory should still be assumed
 * unless known to be mutable.
 */
template<typename T>
kj::Array<kj::ArrayPtr<capnp::word>> coerceSegmentTableToNonConst(T table) {	
	auto outputTable = kj::heapArray<kj::ArrayPtr<capnp::word>>(table.size());
	
	for(size_t i = 0; i < table.size(); ++i) {
		auto maybeConstPtr = table[i].begin();
		capnp::word* nonConstPtr = const_cast<capnp::word*>(maybeConstPtr);
		size_t nWords = table[i].size();
		
		outputTable[i] = kj::ArrayPtr<capnp::word>(nonConstPtr, nWords);
	}
	
	return outputTable;
}

kj::Array<kj::ArrayPtr<const capnp::word>> extractSegmentTable(kj::ArrayPtr<const capnp::word> flatArray) {
	capnp::FlatArrayMessageReader reader(flatArray);
	
	kj::Vector<ArrayPtr<const capnp::word>> segments;
	
	size_t iSegment = 0;
	while(true) {
		auto segment = reader.getSegment(iSegment++);
		if(segment == nullptr)
			break;
		
		segments.add(segment);
	}
	
	return segments.releaseAsArray();
}

kj::Array<cupnp::SegmentTable::Entry> buildSegmentTable(kj::ArrayPtr<kj::ArrayPtr<capnp::word>> input) {
	auto result = kj::heapArrayBuilder<cupnp::SegmentTable::Entry>(input.size());
	for(auto e : input)
		result.add(e);
	return result.finish();
}

}

template<typename T>
struct CupnpMessage {
	// kj::Array<kj::ArrayPtr<capnp::word>> segmentTable;
	kj::Array<cupnp::SegmentTable::Entry> segmentTable;
	
	T root() {
		return cupnp::messageRoot<T>(segmentTable[0], segmentTable);
	}
	
	CupnpMessage(capnp::MessageBuilder& builder) :
		segmentTable(internal::buildSegmentTable(internal::coerceSegmentTableToNonConst(builder.getSegmentsForOutput())))
	{}
	
	CupnpMessage(capnp::MessageReader& reader)
	{
		KJ_REQUIRE(std::is_const<T>::value, "Can only build non-const messages from message builders");
		
		kj::Vector<kj::ArrayPtr<const capnp::word>> segments;
		
		size_t segmentId = 0;
		while(true) {
			auto segment = reader.getSegment(segmentId++);
			
			if(segment == nullptr) break;
			
			segments.add(segment);
		}
		
		segmentTable = internal::buildSegmentTable(internal::coerceSegmentTableToNonConst(segments.releaseAsArray()));
	}
	
	template<typename T2>
	CupnpMessage(Temporary<T2>& t) :
		CupnpMessage(*(t.holder))
	{}
	
	CupnpMessage(kj::ArrayPtr<const kj::ArrayPtr<const capnp::word>> segments) :
		segmentTable(internal::buildSegmentTable(internal::coerceSegmentTableToNonConst(segments)))
	{}
	
	CupnpMessage(kj::ArrayPtr<const capnp::word> flatData) :
		CupnpMessage(internal::extractSegmentTable(flatData))
	{}
	
	template<typename T2>
	CupnpMessage(LocalDataRef<T2>& srcData) :
		CupnpMessage(bytesToWords(srcData.getRaw()))
	{}
};

namespace internal {
	template<typename T, typename T2>
	struct InferCuTIfVoid_ { using type = T2; };
	
	template<typename T>
	struct InferCuTIfVoid_<T, void> { using type = cupnp::CuFor<T>; };
	
	template<typename T1, typename T2>
	using InferCuTIfVoid = typename InferCuTIfVoid_<T1, T2>::type;
}

template<typename CuT = void, typename T>
CupnpMessage<internal::InferCuTIfVoid<T, CuT>> cupnpMessageFromTemporary(Temporary<T>& tmp) {
	return CupnpMessage<internal::InferCuTIfVoid<T, CuT>>(*(tmp.holder));
}
	
template<typename T, typename Device>
struct MapToDevice<CupnpMessage<T>, Device> {
	using Msg = CupnpMessage<T>;
	
	Msg& original;
	Device& device;
	
	kj::Array<MappedData<capnp::word, Device>> deviceSegments;
	
	kj::Array<cupnp::SegmentTable::Entry> hostSegmentTable;
	MappedData<cupnp::SegmentTable::Entry, Device> deviceSegmentTable;
	
	MapToDevice(Msg& original, Device& device) :
		original(original), device(device), deviceSegmentTable(device)
	{
		size_t nSegments = original.segmentTable.size();
		
		// Allocate segments
		{
			auto builder = kj::heapArrayBuilder<MappedData<capnp::word, Device>>(nSegments);
			
			for(size_t i = 0; i < nSegments; ++i) {
				cupnp::SegmentTable::Entry segment = original.segmentTable[i];
				
				builder.add(device, segment.begin(), segment.size());
			}
			
			deviceSegments = builder.finish();
		}
		
		// Gather device pointers
		{
			auto builder = kj::heapArrayBuilder<cupnp::SegmentTable::Entry>(nSegments);
			
			for(size_t i = 0; i < nSegments; ++i) {
				builder.add(kj::ArrayPtr<capnp::word>(
					deviceSegments[i].devicePtr,
					deviceSegments[i].size
				));
			}
			
			hostSegmentTable = builder.finish();
		}
		
		// Map segment table onto device
		deviceSegmentTable = MappedData(device, hostSegmentTable.begin(), hostSegmentTable.size());
		deviceSegmentTable.updateDevice();
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
		kj::ArrayPtr<cupnp::SegmentTable::Entry> ptrSegmentTable(deviceSegmentTable.devicePtr, deviceSegmentTable.size);
		cupnp::SegmentTable::Entry firstSegment = hostSegmentTable[0];
		
		return cupnp::messageRoot<T>(firstSegment, cupnp::SegmentTable(ptrSegmentTable));
	}
};

}