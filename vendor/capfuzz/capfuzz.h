#pragma once

#include <cstdint>

#include <kj/common.h>
#include <kj/async.h>

#include <capnp/dynamic.h>

namespace capfuzz {

/** Input builder to inject custom objects into the protocol calls.
 *
 * When creating capabilities and filling structs, the fuzzer can call methods of this class to 
 * provide the requested data.
 */
struct InputBuilder {
	/**
	 * This class can be used to fill and sub-fields and capabilities as the need arises. It provides
	 * a hook back into the main input filling algorithm which creates the needed struct data and capabilities
	 * based on the buffer filled out by the fuzzer.
	 */
	struct Context {
		virtual void fillStruct(capnp::DynamicStruct::Builder) = 0;
		virtual capnp::DynamicCapability::Client getCapability(capnp::InterfaceSchema) = 0;
		
		inline virtual ~Context() {};
	};
	
	/**
	 * Return a weight that determines how much the internal sampling logic should be biased towards
	 * using this InputBuilder for a given type. 0 means it won't be considered. We recommend making
	 * this number small, as internally the CapFuzz logic collects pointers for builders and uses the
	 * weight as a multiplicity for adding this one to the pool.
	 */
	virtual size_t getWeight(capnp::Type) = 0;
	
	/**
	 * Fill out a given struct. The provided context can be used to fill structs and provide capabilities
	 * as needed.
	 */
	inline virtual void fillStruct(capnp::DynamicStruct::Builder, Context& ctx) { KJ_UNIMPLEMENTED(); };
	
	/**
	 * Create a capability of requested type.
	 */
	inline virtual capnp::Capability::Client getCapability(capnp::InterfaceSchema, Context& ctx) { KJ_UNIMPLEMENTED(); };
	
	inline virtual ~InputBuilder() {};
};

struct ProtocolConfig {
	uint32_t maxListSize = 128;
	uint32_t maxBlobSize = 128;
	
	kj::Array<Own<InputBuilder>> builders = nullptr;
};

/**
 * Run the protocol fuzzing algorithm given a fuzzer input. The fuzzer input will be used to guide the protocol
 * execution until the buffer is exhausted, at which point it will pad it with zeroes.
 *
 * The fuzzing process is deterministic, and built out of the following primitive operations:
 * - Make outbound call
 * - Await outbound call
 * - Cancel outbound call
 * - Wait for an inbound call on a capability created for export in an outgoing call
 * - Fulfill an inbound call previously waited for
 * - Reject an inbound call previously waited for
 * - Drop reference to an imported capability (call result or pipeline)
 * - Create pipeline from outgoing call
 * - Enter sub-object of struct pipeline
 *
 * This method is not intended to test the Cap'n'proto layer, but focuses on the application layer built above.
 * Therefore, types are used for guidance to build correct Cap'n'proto messages.
 */
kj::Promise<void> runFuzzer(kj::ArrayPtr<const kj::byte> data, kj::ArrayPtr<capnp::DynamicCapability::Client> targets, ProtocolConfig config = ProtocolConfig());

}