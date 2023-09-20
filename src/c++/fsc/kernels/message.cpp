#include "message.h"

#include <capnp/any.h>

/*#define CAPNP_PRIVATE
#include <capnp/arena.h>*/

namespace fsc {
	
MessageMappingBase::MessageMappingBase(DeviceBase& device, bool allowAlias) :
	DeviceMappingBase(device),
	allowAlias(allowAlias)
{
}

void MessageMappingBase::updateStructureOnDevice() {
	mapSegments();
	
	kj::Vector<cupnp::SegmentTable::Entry> segmentTableBuilder;
	for(auto& mapping : segmentMappings) {
		segmentTableBuilder.add(mapping -> get());
	}
	segmentTable = mapToDevice(
		segmentTableBuilder.releaseAsArray(),
		*device, true
	);
	
	auto hostTableBuilder = kj::heapArrayBuilder<cupnp::SegmentTable::Entry>(segmentMappings.size());
	for(auto& mapping : segmentMappings) {
		hostTableBuilder.add(mapping -> getHost());
	}
	hostSegmentTable = hostTableBuilder.finish();
}
	
// ---------------- class DeviceMapping<Own<capnp::MessageBuilder>>

DeviceMapping<Own<capnp::MessageBuilder>>::DeviceMapping(
	Own<capnp::MessageBuilder> builder,
	DeviceBase& device,
	bool allowAlias
) :
	MessageMappingBase(device, allowAlias),
	builder(mv(builder))
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

void DeviceMapping<Own<capnp::MessageBuilder>>::mapSegments() {
	auto hostSegments = getHost().getSegmentsForOutput();
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

// ---------------- class DeviceMapping<Own<capnp::MessageReader>>

DeviceMapping<Own<capnp::MessageReader>>::DeviceMapping(
	Own<capnp::MessageReader> reader,
	DeviceBase& device,
	bool allowAlias
) :
	MessageMappingBase(device, allowAlias),
	reader(mv(reader))
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

void DeviceMapping<Own<capnp::MessageReader>>::mapSegments() {
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

}