#pragma once

#include <capnp/common.h>
#include <kj/common.h>
#include <kj/tuple.h>

#define FSC_MVCAP(obj) obj = ::kj::mv(obj)

namespace kj {
	template<typename T>
	class Promise;
	
	template<typename T>
	class Own;
}

namespace fsc {

using byte = kj::byte;
using kj::Promise;
using kj::Own;
using kj::Tuple;
using kj::Maybe;

using kj::mv;
using kj::fwd;
using kj::cp;

using kj::ArrayPtr;
using kj::Array;
using kj::FixedArray;

using kj::instance;


namespace internal {
	
template<typename T>
struct UnwrapMaybe_ {};
template<typename T>
struct UnwrapMaybe_<kj::Maybe<T>> { using Type = T; };

template<typename T>
struct UnwrapIfPromise_ { using Type = T; };
template<typename T>
struct UnwrapIfPromise_<kj::Promise<T>> { using Type = T; };

}

template<typename T>
using UnwrapIfPromise = typename internal::UnwrapIfPromise_<T>::Type;

template<typename T>
using UnwrapMaybe = typename internal::UnwrapMaybe_<T>::Type;

template<typename T>
using ReturnType = decltype(kj::instance<T>()());

inline Array<const byte> wordsToBytes(Array<const word> words) {
	ArrayPtr bytesPtr = words.asBytes();
	return bytesPtr.attach(mv(words));
}

inline Array<byte> wordsToBytes(Array<word> words) {
	ArrayPtr bytesPtr = words.asBytes();
	return bytesPtr.attach(mv(words));
}

struct ID {
	Array<const byte> data;
	
	inline ID() : data(nullptr) {};
	inline ID(const ID& other) : data(kj::heapArray<const byte>(other.data)) {}
	inline ID(ID&& other) : data(mv(other.data)) {}
	
	inline ID(Array<const byte>&& data) : data(mv(data)) {}
	inline ID(const ArrayPtr<const byte>& data) : data(kj::heapArray<const byte>(data)) {}
		
	inline operator ArrayPtr<const byte>() const { return data.asPtr(); }
	inline ArrayPtr<const byte> asPtr() const { return data.asPtr(); }
	
	inline int cmp(const ArrayPtr<const byte>& other) const;
	inline int cmp(const ID& other) const { return cmp(other.data); }
	
	template<typename T>
	inline bool operator <  (const T& other) const { return this->cmp(other) <  0; }
	
	template<typename T>
	inline bool operator <= (const T& other) const { return this->cmp(other) <= 0; }
	
	template<typename T>
	inline bool operator >  (const T& other) const { return this->cmp(other) >  0; }
	
	template<typename T>
	inline bool operator >= (const T& other) const { return this->cmp(other) >= 0; }
	
	template<typename T>
	inline bool operator == (const T& other) const { return this->cmp(other) == 0; }
	
	template<typename T>
	ID fromReader(T) {
		return ID(wordsToBytes(capnp::canonicalize(T)));
	}
};

// === Inline implementation ===

inline int ID::cmp(const ArrayPtr<const byte>& other) const {
		if(data.size() < other.size())
			return -1;
		else if(data.size() > other.size())
			return 1;
	
		return memcmp(data.begin(), other.begin(), data.size());
	}

}