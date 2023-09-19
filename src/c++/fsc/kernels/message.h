#pragma once

#include <cupnp/cupnp.h>
#include "kernels.h"
#include "../data.h"

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

// Mappings for untyped messages

template<>
struct DeviceMapping<Own<capnp::MessageBuilder>> : public DeviceMappingBase {
	DeviceMapping(Own<capnp::MessageBuilder> builder, DeviceBase& device, bool allowAlias);
	~DeviceMapping();
	
	void doUpdateHost() override ;
	void doUpdateDevice() override ;
	
	cupnp::Location get();
	
	capnp::MessageBuilder& getHost();
	
	/** Update the message structure on-device.
	 *
	 * Causes re-allocation of all segments on the device and obtains a new segment table.
	 * \warning Because this method invalidates the result of get(), it may only be used while
	 *          unshared to prevent out-of-memory reads.
	 */
	void updateStructureOnDevice();
	
private:
	bool allowAlias;
	kj::Array<DeviceMappingType<kj::Array<capnp::word>>> segmentMappings;
	DeviceMappingType<kj::Array<cupnp::SegmentTable::Entry>> segmentTable;
	Own<capnp::MessageBuilder> builder;
};

template<>
struct DeviceMapping<Own<capnp::MessageReader>> : public DeviceMappingBase {
	DeviceMapping(Own<capnp::MessageReader> reader, DeviceBase& device, bool allowAlias);
	~DeviceMapping();
	
	void doUpdateHost() override ;
	void doUpdateDevice() override ;
	
	cupnp::Location get();
	
	capnp::MessageReader& getHost();
	
private:
	void updateStructureOnDevice();
	
	bool allowAlias;
	kj::Array<DeviceMappingType<kj::Array<capnp::word>>> segmentMappings;
	DeviceMappingType<kj::Array<cupnp::SegmentTable::Entry>> segmentTable;
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
	CuPtr<CupnpType> get() { return CuPtr<CupnpType>(getUntyped()); }
	
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
	CuPtr<const CupnpType> get() { return CuPtr<const CupnpType>(getUntyped()); }
	
	capnp::MessageReader& getHostUntyped() { return Parent::getHost(); }
	typename HostType::Reader getHost() { return Parent::getHost().template getRoot<HostType>(); }
};

}