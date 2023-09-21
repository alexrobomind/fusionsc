#pragma once

#include <cstdint>

#include <kj/common.h>
#include <kj/async.h>

#include <capnp/dynamic.h>

namespace capfuzz {

struct InputBuilder {
	struct Context {
		virtual void fillStruct(capnp::DynamicStruct::Builder) = 0;
		virtual capnp::DynamicCapability::Client getCapability(capnp::InterfaceSchema) = 0;
		
		inline virtual ~Context() {};
	};
	
	virtual size_t getWeight(capnp::Type) = 0;
	
	inline virtual void fillStruct(capnp::DynamicStruct::Builder, Context& ctx) { KJ_UNIMPLEMENTED(); };
	inline virtual capnp::Capability::Client getCapability(capnp::InterfaceSchema, Context& ctx) { KJ_UNIMPLEMENTED(); };
	
	inline virtual ~InputBuilder() {};
};

struct ProtocolConfig {
	uint32_t maxListSize = 128;
	uint32_t maxBlobSize = 128;
	
	kj::ArrayPtr<InputBuilder*> builders = nullptr;
};

kj::Promise<void> runFuzzer(kj::ArrayPtr<const kj::byte> data, kj::ArrayPtr<capnp::DynamicCapability::Client> targets, ProtocolConfig config = ProtocolConfig());

}