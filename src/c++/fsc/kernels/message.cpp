#include "cudata.h"

#include <capnp/any.h>

/*#define CAPNP_PRIVATE
#include <capnp/arena.h>*/

namespace fsc {
	
// ---------------- class DeviceMapping<Own<capnp::MessageBuilder>>

DeviceMapping<Own<capnp::MessageBuilder>>::DeviceMapping(
	Own<capnp::MessageBuilder> builder,
	DeviceBase& device,
	bool allowAlias
) :
	DeviceMappingBase(device),
	builder(mv(builder)),
	allowAlias(allowAlias)
{
	updateStructureOnDevice();
}

DeviceMapping<Own<capnp::MessageBuilder>>::~DeviceMapping() {
}

void DeviceMapping<Own<capnp::MessageBuilder>>::doUpdateHost() {
	for(auto& mapping : segmentMappings)
		mapping -> doUpdateHost();
}

void DeviceMapping<Own<capnp::MessageBuilder>>::doUpdateDevice() {
	for(auto& mapping : segmentMappings)
		mapping -> doUpdateDevice();
}

capnp::MessageBuilder& DeviceMapping<Own<capnp::MessageBuilder>>::getHost() {
	return *builder;
}

cupnp::Location DeviceMapping<Own<capnp::MessageBuilder>>::get() {
	cupnp::Location result;
	result.segmentId = 0;
	result.ptr = reinterpret_cast<kj::byte*>(segmentMappings[0] -> get().begin());
	result.segments = segmentTable -> get();
	
	return result;
}

void DeviceMapping<Own<capnp::MessageBuilder>>::updateStructureOnDevice() {
	// using capnp::_::SegmentId;
	
	// Get access to the underlying message
	/*capnp::AnyPointer::Builder root = getHost().getRoot<capnp::AnyPointer>();
	capnp::_::PointerBuilder rootInternal = capnp::_::PointerHelpers<capnp::AnyPointer>
		::getInternalBuilder(cp(root));
	capnp::_::BuilderArena& arena = *(rootInternal.getArena());
	
	// Map segments
	{
		kj::Vector<DeviceMappingType<kj::Array<capnp::word>>> segmentMappingBuilder;
		for(unsigned int i = 0; arena.tryGetSegment(SegmentId(i)) != nullptr; ++i) {
			capnp::_::SegmentBuilder* segment = arena.getSegment(SegmentId(i));
			KJ_REQUIRE(segment -> isWritable(), "Can not map segments added using Orphanage::referenceExternalData");
			
			capnp::word* start = segment -> getPtrUnchecked(0 * capnp::WORDS);
			size_t nWords = segment -> getSize() / capnp::WORDS;
			segmentMappingBuilder.add(mapToDevice(
				kj::ArrayPtr<capnp::word>(start, nWords).attach(nullptr),
				*device, true
			));
		}
		segmentMappings = segmentMappingBuilder.releaseAsArray();
	}*/
	auto hostSegments = getHost().getSegmentsForOutput();
	{
		kj::Vector<DeviceMappingType<kj::Array<capnp::word>>> segmentMappingBuilder;
		for(kj::ArrayPtr<const capnp::word> hostSegment : hostSegments) {
			capnp::word* start = const_cast<capnp::word*>(hostSegment.begin());
			segmentMappingBuilder.add(mapToDevice(
				kj::ArrayPtr<capnp::word>(start, hostSegment.size()).attach(nullptr),
				*device, true
			));
		}
		segmentMappings = segmentMappingBuilder.releaseAsArray();
	}
	
	
	// Create segment table
	{
		kj::Vector<cupnp::SegmentTable::Entry> segmentTableBuilder;
		for(auto& mapping : segmentMappings) {
			segmentTableBuilder.add(mapping -> get());
		}
		segmentTable = mapToDevice(
			segmentTableBuilder.releaseAsArray(),
			*device, true
		);
	}
}

// ---------------- class DeviceMapping<Own<capnp::MessageReader>>

DeviceMapping<Own<capnp::MessageReader>>::DeviceMapping(
	Own<capnp::MessageReader> reader,
	DeviceBase& device,
	bool allowAlias
) :
	DeviceMappingBase(device),
	reader(mv(reader)),
	allowAlias(allowAlias)
{
	updateStructureOnDevice();
}

DeviceMapping<Own<capnp::MessageReader>>::~DeviceMapping() {
}

void DeviceMapping<Own<capnp::MessageReader>>::doUpdateHost() {
	KJ_FAIL_REQUIRE("Updating the host is unsupported on a message reader");
}

void DeviceMapping<Own<capnp::MessageReader>>::doUpdateDevice() {
	for(auto& mapping : segmentMappings)
		mapping -> doUpdateDevice();
}

capnp::MessageReader& DeviceMapping<Own<capnp::MessageReader>>::getHost() {
	return *reader;
}

cupnp::Location DeviceMapping<Own<capnp::MessageReader>>::get() {
	cupnp::Location result;
	result.segmentId = 0;
	result.ptr = reinterpret_cast<kj::byte*>(segmentMappings[0] -> get().begin());
	result.segments = segmentTable -> get();
	
	return result;
}

void DeviceMapping<Own<capnp::MessageReader>>::updateStructureOnDevice() {
	// using capnp::_::SegmentId;
	
	// Get access to the underlying message
	/*capnp::AnyPointer::Reader root = getHost().getRoot<capnp::AnyPointer>();
	capnp::_::PointerReader rootInternal = capnp::_::PointerHelpers<capnp::AnyPointer>
		::getInternalReader(cp(root));
	
	// Map segments
	KJ_IF_MAYBE(pArena, rootInternal.getArena()) {
		capnp::_::Arena& arena = *pArena;
		
		kj::Vector<DeviceMappingType<kj::Array<capnp::word>>> segmentMappingBuilder;
		for(unsigned int i = 0; arena.tryGetSegment(SegmentId(i)) != nullptr; ++i) {
			capnp::_::SegmentReader* segment = arena.tryGetSegment(SegmentId(i));
			
			capnp::word* start = const_cast<capnp::word*>(
				segment -> getStartPtr()
			);
			size_t nWords = segment -> getSize() / capnp::WORDS;
			segmentMappingBuilder.add(mapToDevice(
				kj::ArrayPtr<capnp::word>(start, nWords).attach(nullptr),
				*device, true
			));
		}
		segmentMappings = segmentMappingBuilder.releaseAsArray();
	} else {
		KJ_FAIL_REQUIRE("Reader root had no builder arena. This should not happen.");
	}*/
	{
		kj::Vector<DeviceMappingType<kj::Array<capnp::word>>> segmentMappingBuilder;
		for(unsigned int iSegment = 0;;++iSegment) {
			kj::ArrayPtr<const capnp::word> hostSegment = getHost().getSegment(iSegment);
			
			if(hostSegment == nullptr)
				break;
			
			capnp::word* start = const_cast<capnp::word*>(hostSegment.begin());
			segmentMappingBuilder.add(mapToDevice(
				kj::ArrayPtr<capnp::word>(start, hostSegment.size()).attach(nullptr),
				*device, true
			));
		}
		segmentMappings = segmentMappingBuilder.releaseAsArray();
	}
	
	// Create segment table
	{
		kj::Vector<cupnp::SegmentTable::Entry> segmentTableBuilder;
		for(auto& mapping : segmentMappings) {
			segmentTableBuilder.add(mapping -> get());
		}
		segmentTable = mapToDevice(
			segmentTableBuilder.releaseAsArray(),
			*device, true
		);
	}
}

}