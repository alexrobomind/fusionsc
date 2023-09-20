#pragma once

#include <cupnp/cupnp.h>
#include "kernels.h"
#include "array.h"
#include "../data.h"

#include <capnp/serialize.h>

namespace fsc {
		
namespace internal {

inline kj::Array<const capnp::word> newEmptyMessage() {
	auto emptyMessage = kj::heapArray<kj::byte>({
		// Remember: capnp storage specification is ALWAYS LITTLE ENDIAN
		// 4-byte segment count (minus one), so 0 since there is one segment
		0, 0, 0, 0,
		// 4-byte segment size (in 8-byte words)
		1, 0, 0, 0,
		// 8-byte segment
		0, 0, 0, 0, 0, 0, 0, 0
	});
	return bytesToWords(mv(emptyMessage));
}

inline kj::Array<const capnp::word> EMPTY_MESSAGE = internal::newEmptyMessage();

}

// Helpers to attach types to message builders & readers

template<typename HostType, typename CupnpType>
struct CuTypedMessageBuilder {
	Own<capnp::MessageBuilder> builder;
};

template<typename HostType, typename CupnpType>
struct CuTypedMessageReader {
	Own<capnp::MessageReader> reader;
};

#define FSC_READER_MAPPING(NS, Type) DeviceMappingType<CuTypedMessageReader<NS::Type, NS::cu::Type>>
#define FSC_BUILDER_MAPPING(NS, Type) DeviceMappingType<CuTypedMessageBuilder<NS::Type, NS::cu::Type>>

template<typename HostType, typename CupnpType>
CuTypedMessageBuilder<HostType, CupnpType> cuBuilder(Own<capnp::MessageBuilder> builder) {
	return { mv(builder) };
}

template<typename HostType, typename CupnpType>
CuTypedMessageBuilder<HostType, CupnpType> cuBuilder(Temporary<HostType>&& tmp) {
	return cuBuilder<HostType, CupnpType>(mv(tmp.holder));
}

template<typename HostType, typename CupnpType>
CuTypedMessageReader<HostType, CupnpType> cuReader(Own<capnp::MessageReader> reader) {
	return { mv(reader) };
}

template<typename HostType, typename CupnpType>
CuTypedMessageReader<HostType, CupnpType> cuReader(LocalDataRef<HostType> ldr) {
	Own<capnp::MessageReader> reader = kj::heap<capnp::FlatArrayMessageReader>(bytesToWords(ldr.getRaw())).attach(ldr);
	return cuReader<HostType, CupnpType>(mv(reader));
}

template<typename HostType, typename CupnpType>
CuTypedMessageReader<HostType, CupnpType> cuReader(std::nullptr_t) {
	Own<capnp::MessageReader> reader = kj::heap<capnp::FlatArrayMessageReader>(internal::EMPTY_MESSAGE);
	return cuReader<HostType, CupnpType>(mv(reader));
}

template<typename HostType, typename CupnpType, typename T>
CuTypedMessageReader<HostType, CupnpType> cuReader(Maybe<T> maybe) {
	KJ_IF_MAYBE(pVal, maybe) {
		return cuReader<HostType, CupnpType>(mv(*pVal));
	} else {
		return cuReader<HostType, CupnpType>(nullptr);
	}
}

#define FSC_MAP_READER(NS, Type, reader, device, allowAlias) mapToDevice(cuReader<NS::Type, NS::cu::Type>(reader), device, allowAlias)
#define FSC_MAP_BUILDER(NS, Type, reader, device, allowAlias) mapToDevice(cuBuilder<NS::Type, NS::cu::Type>(reader), device, allowAlias)

// Mappings for untyped messages

struct MessageMappingBase : public DeviceMappingBase {
	MessageMappingBase(DeviceBase& device, bool allowAlias);
	
	void updateStructureOnDevice();
	
	const bool allowAlias;
	
	template<typename T>
	T getRoot();
	
	template<typename T>
	T getHostRoot();
	
protected:
	virtual void mapSegments() = 0;
	kj::Array<DeviceMappingType<kj::Array<capnp::word>>> segmentMappings;
	kj::Array<cupnp::SegmentTable::Entry> hostSegmentTable;
	DeviceMappingType<kj::Array<cupnp::SegmentTable::Entry>> segmentTable;
};

template<>
struct DeviceMapping<Own<capnp::MessageBuilder>> : public MessageMappingBase {
	DeviceMapping(Own<capnp::MessageBuilder> builder, DeviceBase& device, bool allowAlias);
	~DeviceMapping();
	
	void doUpdateHost() override ;
	void doUpdateDevice() override ;
	
	cupnp::Location get();
	
	capnp::MessageBuilder& getHost();
	
private:
	void mapSegments() override;
	
	Own<capnp::MessageBuilder> builder;
};

template<>
struct DeviceMapping<Own<capnp::MessageReader>> : public MessageMappingBase {
	DeviceMapping(Own<capnp::MessageReader> reader, DeviceBase& device, bool allowAlias);
	~DeviceMapping();
	
	void doUpdateHost() override ;
	void doUpdateDevice() override ;
	
	cupnp::Location get();
	
	capnp::MessageReader& getHost();
	
private:
	void mapSegments() override;
	
	Own<capnp::MessageReader> reader;
};

// Mappings for typed messages

template<typename HostType, typename CupnpType>
struct DeviceMapping<CuTypedMessageBuilder<HostType, CupnpType>> : public DeviceMapping<Own<capnp::MessageBuilder>> {
	using Parent = DeviceMapping<Own<capnp::MessageBuilder>>;
	
	DeviceMapping(CuTypedMessageBuilder<HostType, CupnpType>&& typedBuilder, DeviceBase& device, bool allowAlias) :
		Parent(mv(typedBuilder.builder), device, allowAlias)
	{}
	
	cupnp::Location getUntyped() { return Parent::get(); }
	typename CupnpType::Builder get() { return getRoot<typename CupnpType::Builder>(); }
	
	capnp::MessageBuilder& getHostUntyped() { return Parent::getHost(); }
	typename HostType::Builder getHost() { return Parent::getHost().template getRoot<HostType>(); }
};

template<typename HostType, typename CupnpType>
struct DeviceMapping<CuTypedMessageReader<HostType, CupnpType>> : public DeviceMapping<Own<capnp::MessageReader>> {
	using Parent = DeviceMapping<Own<capnp::MessageReader>>;
	
	DeviceMapping(CuTypedMessageReader<HostType, CupnpType>&& typedReader, DeviceBase& device, bool allowAlias) :
		Parent(mv(typedReader.reader), device, allowAlias)
	{}
	
	cupnp::Location getUntyped() { return Parent::get(); }
	typename CupnpType::Reader get() { return getRoot<typename CupnpType::Reader>(); }
	
	capnp::MessageReader& getHostUntyped() { return Parent::getHost(); }
	typename HostType::Reader getHost() { return Parent::getHost().template getRoot<HostType>(); }
};

template<typename T>
T MessageMappingBase::getRoot() {	
	// Inspect message to read host pointer
	cupnp::Location loc;
	loc.segmentId = 0;
	loc.ptr = reinterpret_cast<kj::byte*>(segmentMappings[0] -> getHost().begin());
	loc.segments = hostSegmentTable.asPtr();
	
	auto decoded = cupnp::getPointer<cupnp::AnyPointer::Builder>(loc);
	
	// Translate host pointer to device
	auto segmentId = decoded.data.segmentId;
	auto hostStart = reinterpret_cast<kj::byte*>(segmentMappings[segmentId] -> getHost().begin());
	auto deviceStart = reinterpret_cast<kj::byte*>(segmentMappings[segmentId] -> get().begin());
	
	cupnp::Location deviceData;
	deviceData.segmentId = segmentId;
	deviceData.ptr = deviceStart + (decoded.data.ptr - hostStart);
	deviceData.segments = segmentTable -> get();
	
	return T(decoded.structure, deviceData);
}

template<typename T>
T MessageMappingBase::getHostRoot() {	
	// Inspect message to read host pointer
	cupnp::Location loc;
	loc.segmentId = 0;
	loc.ptr = reinterpret_cast<kj::byte*>(segmentMappings[0] -> getHost().begin());
	loc.segments = hostSegmentTable.asPtr();
	
	return cupnp::getPointer<T>(loc);
}

}