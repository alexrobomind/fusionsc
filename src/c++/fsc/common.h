#pragma once

#ifdef DOXYGEN
#define FSC_WITH_CUDA
#endif

#include <capnp/common.h>
#include <capnp/message.h>
#include <kj/common.h>
#include <kj/tuple.h>

#include <kj/async.h>
#include <kj/debug.h>

#define FSC_MVCAP(obj) obj = ::kj::mv(obj)

#define FSC_REQUIRE_MAYBE(result, maybe, ...) \
	auto result = kj::_::readMaybe(maybe); \
	KJ_REQUIRE(result, __VA_ARGS__);

#define FSC_ASSERT_MAYBE(result, maybe, ...) \
	auto result = kj::_::readMaybe(maybe); \
	KJ_ASSERT(result, __VA_ARGS__);

#define FSC_MAYBE_OR_RETURN(result, maybe, returnExpr) \
	auto result = kj::_::readMaybe(maybe); \
	if(!result) { return returnExpr; }

namespace kj {
	template<typename T>
	class Promise;
	
	/*template<typename T, typename StaticDisposer>
	class Own;*/
}

/**
 * The FSC library.
 */
namespace fsc {
		
constexpr inline double pi = 3.14159265358979323846; // "Defined" in magnetics.cpp

using byte = kj::byte;
using kj::Promise;
using kj::ForkedPromise;
using kj::Own;
using kj::Tuple;
using kj::Maybe;
using kj::OneOf;
using kj::Decay;
using kj::RemoveConst;

using kj::mv;
using kj::fwd;
using kj::cp;
using kj::tuple;
using kj::refTuple;

using kj::ArrayPtr;
using kj::Array;
using kj::FixedArray;

using kj::instance;

using kj::READY_NOW;
using kj::NEVER_DONE;

using kj::PromiseFulfiller;
using kj::CrossThreadPromiseFulfiller;


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

/*
//! Maps Promise<T> to T, otherwise returns T, no recursive unpacking
template<typename T>
using UnwrapIfPromise = typename internal::UnwrapIfPromise_<T>::Type;

//! Wraps Maybe<T> to T, otherwise returs T, no recursive unpacking
template<typename T>
using UnwrapMaybe = typename internal::UnwrapMaybe_<T>::Type;

//! The result type of calling T(Args) with instances
template<typename T, typename... Args>
using ReturnType = decltype(kj::instance<T>(kj::instance<Args>()...));*/

//! Casts capnp word (8 bytes) array to byte array
inline Array<const byte> wordsToBytes(Array<const capnp::word> words) {
	ArrayPtr<const byte> bytesPtr = words.asBytes();
	return bytesPtr.attach(mv(words));
}

//! Casts capnp word (8 bytes) array to byte array
inline Array<byte> wordsToBytes(Array<capnp::word> words) {
	ArrayPtr<byte> bytesPtr = words.asBytes();
	return bytesPtr.attach(mv(words));
}

//! Casts byte array to capnp word (8 bytes) array. Undefined if not aligned.
inline Array<const capnp::word> bytesToWords(Array<const byte> bytes) {
	ArrayPtr<const capnp::word> wordPtr = ArrayPtr<const capnp::word>(
		reinterpret_cast<const capnp::word*>(bytes.begin()),
		bytes.size() / sizeof(capnp::word)
	);
	return wordPtr.attach(mv(bytes));
}

//! Casts byte array to capnp word (8 bytes) array. Undefined if not aligned.
inline ArrayPtr<const capnp::word> bytesToWords(ArrayPtr<const byte> bytes) {
	ArrayPtr<const capnp::word> wordPtr = ArrayPtr<const capnp::word>(
		reinterpret_cast<const capnp::word*>(bytes.begin()),
		bytes.size() / sizeof(capnp::word)
	);
	return wordPtr;
}

//! Casts byte array to capnp word (8 bytes) array. Undefined if not aligned.
inline Array<capnp::word> bytesToWords(Array<byte> bytes) {
	ArrayPtr<capnp::word> wordPtr = ArrayPtr<capnp::word>(
		reinterpret_cast<capnp::word*>(bytes.begin()),
		bytes.size() / sizeof(capnp::word)
	);
	return wordPtr.attach(mv(bytes));
}

/*
//! Type of a kj::tuple constructed from given types.
template<typename... T>
using TupleFor = decltype(tuple(instance<T>()...));

// Similar to kj::_::Void (which is, annoyingly, internal), a placeholder for "nothing"
// We don't call it Void both to maintain compatibility and because of semantic differences

//! Replacement type for void in tuples created by joinPromises()
struct Meaningless {};

template<typename T> struct VoidToMeaningless_       { using Type = T; };
template<>           struct VoidToMeaningless_<void> { using Type = Meaningless; };

//! Replaces void by Meaningless
template<typename T> using VoidToMeaningless = typename VoidToMeaningless_<T>::Type;

template<typename T>
struct TVoid_ { using Type = void; };

//! Non-deduced void
template<typename T>
using TVoid = typename TVoid_<T>::Type;

// Join promises into a tuple

//! Joins promises into a tuple. Promises to void are replaced with an instance of Meaningless in the result.
template<typename T1, typename... T>
Promise<TupleFor<T1, VoidToMeaningless<T>...>> joinPromises(Promise<T1>&& p1, Promise<T>&&... tail) {
	Tuple<Promise<T>...> tailTuple = tuple(mv(tail)...);
	
	return p1.then([tailTuple = mv(tailTuple)](T1 t1) -> Promise<TupleFor<T1, VoidToMeaningless<T>...>> {
		return kj::apply(joinPromises, mv(tailTuple)).then([t1 = mv(t1)](TupleFor<VoidToMeaningless<T>...> t) -> TupleFor<T1, VoidToMeaningless<T>...> {
			return tuple(t1, t);
		});
	});
}

template<typename... T>
Promise<TupleFor<Meaningless, VoidToMeaningless<T>...>> joinPromises(Promise<void>&& p1, Promise<T>&&... tail) {
	Tuple<Promise<T>...> tailTuple = tuple(mv(tail)...);
	
	return p1.then([tailTuple = mv(tailTuple)]() -> Promise<TupleFor<Meaningless, VoidToMeaningless<T>...>> {
		return kj::apply(joinPromises, mv(tailTuple)).then([](TupleFor<VoidToMeaningless<T>...> t) -> TupleFor<Meaningless, VoidToMeaningless<T>...> {
			return tuple(Meaningless(), t);
		});
	});
}

inline Promise<Tuple<>> joinPromises() { return tuple(); }*/

//! Identifier class wrapping a byte array
struct ID {
	Array<const byte> data;
	
	inline ID() : data(nullptr) {};
	inline ID(const ID& other) : data(kj::heapArray<const byte>(other.data)) {}
	inline ID(ID&& other) : data(mv(other.data)) {}
	
	inline ID& operator=(const ID& other) { data = kj::heapArray<const byte>(other.data); return *this; }
	inline ID& operator=(ID&& other) { data = mv(other.data); return *this; }
	
	inline ID(const ArrayPtr<const byte>& data) : data(kj::heapArray<const byte>(data)) {}
		
	inline operator ArrayPtr<const byte>() const { return data.asPtr(); }
	inline ArrayPtr<const byte> asPtr() const { return data.asPtr(); }
	
	inline int cmp(const ArrayPtr<const byte>& other) const;
	inline int cmp(const ID& other) const { return cmp(other.data); }
	inline int cmp(decltype(nullptr)) const { return cmp(ID()); }
	
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
	inline bool operator != (const T& other) const { return this->cmp(other) != 0; }
	
	//! Construct ID by from Reader. Requires data.h
	/** This method constructs an ID out of the canonical representation of the passed
	 *  capnproto reader.
	 
	 *  If the reader holds any capabilities (such as DataRef),
	 *  the canonicalization will fail. Use fromReaderWithRefs() instead.
	 *
	 * \note Requires data.h
	 */
	template<typename T>
	static ID fromReader(T t);
	
	//! Construct ID from reader with datarefs. Requires data.h
	/** In addition to fromReader(), this method also replaces all
	 *  linked DataRef objects with their IDs. Since this might
	 *  require remote calls, it can only return a Promise to
	 *  an ID.
	 *
	 * \note Requires data.h
	 */
	template<typename T>
	static Promise<ID> fromReaderWithRefs(T t);
};

// === Inline implementation ===

inline int ID::cmp(const ArrayPtr<const byte>& other) const {
		if(data.size() < other.size())
			return -1;
		else if(data.size() > other.size())
			return 1;
		
		if(data.size() == 0)
			return 0;
	
		return memcmp(data.begin(), other.begin(), data.size());
	}

}
