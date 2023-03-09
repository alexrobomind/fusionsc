#pragma once

#include <cupnp/cupnp.h>
#include "kernels.h"
#include "data.h"

#include <capnp/serialize.h>

namespace fsc {
	
template<typename T>
using CuPtr = cupnp::TypedLocation<T>;
	
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

template<typename HostType, typename CupnpType>
auto cuTyped(Own<capnp::MessageBuilder> builder) {
	return CuTypedMessageBuilder<HostType, CupnpType> { mv(builder); }
}

template<typename HostType, typename CupnpType>
auto cuTyped(Own<capnp::MessageReader> builder) {
	return CuTypedMessageReader<HostType, CupnpType> { mv(builder); }
}

template<typename HostType, typename CupnpType>
auto cuTyped(LocalDataRef<HostType> ldr) {
	Own<capnp::MessageReader> reader = kj::heap<capnp::FlatArrayMessageReader>(ldr.getRaw()).attach(ldr);
	return cuTyped<HostType, CupnpType>(mv(reader));
}

template<typename HostType, typename CupnpType>
auto cuTyped(std::nullptr_t) {
	Own<capnp::MessageReader> reader = kj::heap<capnp::FlatArrayMessageReader>(internal::EMPTY_MESSAGE);
	return cuTyped<HostType, CupnpType>(mv(reader));
}

// Mappings for untyped messages

template<>
struct DeviceMapping<Own<capnp::MessageBuilder>> : public DeviceMappingBase {
	DeviceMapping(Own<capnp::MessageBuilder> builder, DeviceBase& device, bool allowAlias);
	~DeviceMapping();
	
	void doUpdateHost() override ;
	void doUpdateDevice() override ;
	
	cupnp::Location get();
	
private:
	kj::Array<Own<DeviceMapping<kj::Array<kj::word>>>> segmentMappings;
};

template<>
struct DeviceMapping<Own<capnp::MessageReader>> : public DeviceMappingBase {
	DeviceMapping(Own<capnp::MessageReader> builder, DeviceBase& device, bool allowAlias);
	~DeviceMapping();
	
	void doUpdateHost() override ;
	void doUpdateDevice() override ;
	
	cupnp::Location get();
	
private:
	kj::Array<Own<DeviceMapping<kj::Array<kj::word>>>> segmentMappings;
};

// Mappings for typed messages

template<typename HostType, typename CupnpType>
struct DeviceMapping<CuTypedMessageBuilder<HostType, CupnpType>> : public DeviceMapping<Own<MessageBuilder>> {
	DeviceMapping(CuTypedMessageBuilder<HostType, CupnpType>&& typedBuilder, DeviceBase& device, bool allowAlias) :
		DeviceMapping<Own<MessageBuilder>>(mv(typedBuilder.builder), device, allowAlias)
	{}
	
	cupnp::Location getUntyped() { return DeviceMapping<Own<MessageBuilder>>::get(); }
	CuPtr<CupnpType> get() { return getUntyped(); }
};

template<typename HostType, typename CupnpType>
struct DeviceMapping<CuTypedMessageReader<HostType, CupnpType>> : public DeviceMapping<Own<MessageReader>> {
	DeviceMapping(CuTypedMessageReader<HostType, CupnpType>&& typedReader, DeviceBase& device, bool allowAlias) :
		DeviceMapping<Own<MessageReader>>(mv(typedReader.reader), device, allowAlias)
	{}
	
	cupnp::Location getUntyped() { return DeviceMapping<Own<MessageReader>>::get(); }
	CuPtr<CupnpType> get() { return getUntyped(); }
};

}