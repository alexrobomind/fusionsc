#pragma once

#include <cstdint>

namespace goldfish
{
	inline uint16_t from_big_endian(uint16_t x) { return _byteswap_ushort(x); }
	inline uint32_t from_big_endian(uint32_t x) { return _byteswap_ulong(x); }
	inline uint64_t from_big_endian(uint64_t x) { return _byteswap_uint64(x); }

	inline uint16_t to_big_endian(uint16_t x) { return from_big_endian(x); }
	inline uint32_t to_big_endian(uint32_t x) { return from_big_endian(x); }
	inline uint64_t to_big_endian(uint64_t x) { return from_big_endian(x); }
}