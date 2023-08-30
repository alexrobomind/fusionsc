#pragma once

#include <cstdint>

#include <kj/common.h>
#include <kj/async.h>

#include <capnp/dynamic.h>

namespace capfuzz {

struct ProtocolConfig {
	uint32_t maxListSize = 128;
	uint32_t maxBlobSize = 128;
};

kj::Promise<void> runFuzzer(kj::ArrayPtr<const kj::byte> data, kj::ArrayPtr<capnp::DynamicCapability::Client> targets, ProtocolConfig config = ProtocolConfig());

}