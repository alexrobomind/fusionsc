#pragma once

#include <cstdint>

namespace gold_fish
{
	inline uint16_t byte_swap(uint16_t x) { return _byteswap_ushort(x); }
	inline uint32_t byte_swap(uint32_t x) { return _byteswap_ulong(x); }
	inline uint64_t byte_swap(uint64_t x) { return _byteswap_uint64(x); }
}