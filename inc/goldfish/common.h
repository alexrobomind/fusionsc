#pragma once

#include <cstdint>
#include <iterator>
#include <stdlib.h>

namespace goldfish
{
	using byte = uint8_t;

	inline uint16_t from_big_endian(uint16_t x) { return _byteswap_ushort(x); }
	inline uint32_t from_big_endian(uint32_t x) { return _byteswap_ulong(x); }
	inline uint64_t from_big_endian(uint64_t x) { return _byteswap_uint64(x); }

	inline uint16_t to_big_endian(uint16_t x) { return from_big_endian(x); }
	inline uint32_t to_big_endian(uint32_t x) { return from_big_endian(x); }
	inline uint64_t to_big_endian(uint64_t x) { return from_big_endian(x); }

	// All goldfish exceptions subclass this exception
	struct exception {};

	// Base class for all formatting errors that happen while parsing a document
	struct ill_formatted : exception {};

	// Specifically for IO errors, thrown by file_reader/writer and istream_reader/writer
	struct io_exception : exception {};
	struct io_exception_with_error_code : io_exception
	{
		io_exception_with_error_code(int _error_code)
			: error_code(_error_code)
		{}

		int error_code;
	};


	// VC++ has a make_unchecked_array_iterator API to allow using raw iterators in APIs like std::copy or std::equal
	// We implement our own that forwards to VC++ implementation or is identity depending on the compiler
	template <class T> auto make_unchecked_array_iterator(T&& t) { return stdext::make_unchecked_array_iterator(std::forward<T>(t)); }
	template <class T> auto get_array_iterator_from_unchecked(T&& t) { return t.base(); }
}