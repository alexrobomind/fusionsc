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

}