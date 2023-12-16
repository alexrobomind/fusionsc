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
	class exception : public std::exception
	{
	public:
		exception(const char* w)
			: m_what(w)
		{}
		const char* what() const noexcept override { return m_what; }
	private:
		const char* m_what;
	};

	// Base class for all formatting errors that happen while parsing a document
	struct ill_formatted : exception { using exception::exception; };

	// Specifically for IO errors, thrown by file_reader/writer and istream_reader/writer
	struct io_exception : exception { using exception::exception; };
	struct io_exception_with_error_code : io_exception
	{
		io_exception_with_error_code(const char* w, int _error_code)
			: io_exception(w), error_code(_error_code)
		{}

		int error_code;
	};

	// CBOR supports an "undefined" type, which is represented at runtime by the C++ type below
	struct undefined {};
	inline bool operator == (const undefined&, const undefined&) { return true; }
	inline bool operator < (const undefined&, const undefined&) { return false; }

	// VC++ has a make_unchecked_array_iterator API to allow using raw iterators in APIs like std::copy or std::equal
	// We implement our own that forwards to VC++ implementation or is identity depending on the compiler
	template <class T> auto make_unchecked_array_iterator(T&& t) { return stdext::make_unchecked_array_iterator(std::forward<T>(t)); }
	template <class T> auto get_array_iterator_from_unchecked(T&& t) { return t.base(); }

	template <size_t...> struct largest {};
	template <size_t x> struct largest<x> { enum { value = x }; };
	template <size_t x, size_t y> struct largest<x, y> { enum { value = x > y ? x : y }; };
	template <size_t x, size_t y, size_t... z> struct largest<x, y, z...> { enum { value = largest<x, largest<y, z...>::value>::value }; };

	template <class...> struct conjunction {};
	template <> struct conjunction<> { enum { value = true }; };
	template <class Head, class... Tail> struct conjunction<Head, Tail...> { enum { value = Head::value && conjunction<Tail...>::value }; };

	template <class...> struct disjunction {};
	template <> struct disjunction<> { enum { value = false }; };
	template <class Head, class... Tail> struct disjunction<Head, Tail...> { enum { value = Head::value || disjunction<Tail...>::value }; };

	// Used for all cases where we need to read from a stream in chunks (default implementation of seek, stream copy, etc...)
	static const int typical_buffer_length = 8 * 1024;
}