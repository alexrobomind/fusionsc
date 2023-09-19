#pragma once

#include "common.h"

namespace fsc {

/**
 * A pointer-like with special ownership semantics. Invoking its move constructor
 * will transfer ownership of the contained object, while invoking the move
 * constructor will create a non-owning pointer. To reduce the probability of a
 * segmentation fault, the owning part must be explicitly released.
 *
 * This is mostly useful for lambdas, as Held instances can be passed into lambdas
 * by value like pointers, while the owning part can then later be attached somewhere
 * else. This is a more efficient alternative to shared pointers.
 */
template<typename T>
struct Held {
	Held(Own<T>&& src) :
		owningPtr(mv(src)),
		ref(*owningPtr)
	{}
	
	Held(const Held& other) :
		owningPtr(),
		ref(other.ref)
	{}
	
	Held(Held&& other) = default;
	
	~Held() {
		if(!ud.isUnwinding()) {
			KJ_REQUIRE(owningPtr.get() == nullptr, "Destroyed Held<...> without ownership transfer");
		} else {
			if(owningPtr.get() != nullptr) {
				KJ_LOG(WARNING, "Unwinding across a Held<...>. Application might segfault");
			}
		}
	}
	
	T& operator*() { return ref; }
	T* operator->() { return &ref; }
	T* get() { return &ref; }
	
	template<typename... Params>
	void attach(Params&&... params) {
		owningPtr = owningPtr.attach(fwd<Params>(params)...);
	}
	
	Own<T> release() { KJ_REQUIRE(owningPtr.get() == &ref, "Releasing already-released held"); return mv(owningPtr); }
	Own<T> x() { return release(); }
	
private:
	Own<T> owningPtr;
	T& ref;
	kj::UnwindDetector ud;
};

template<typename T, typename... Params>
Held<T> heapHeld(Params&&... params) {
	return Held<T>(kj::heap<T>(fwd<Params>(params)...));
}

template<typename T>
Held<T> ownHeld(Own<T>&& src) {
	return Held<T>(mv(src));
}

template<typename T>
struct Shared {
	using Payload = T;
	
	template<typename... Params>
	Shared(Params&&... t) :
		impl(kj::refcounted<Impl>(kj::fwd<Params>(t)...))
	{}
	
	Shared(const Shared<T>& other) :
		impl(other.impl -> addRef())
	{}
	
	Shared(Shared<T>& other) :
		impl(other.impl -> addRef())
	{}
	
	Shared<T>& operator=(const Shared<T>& other) {
		impl = other.impl -> addRef();
	}
	
	Shared(Shared<T>&& other) = default;
	Shared<T>& operator=(Shared<T>&& other) = default;
	
	T& get() { return *(impl -> payload); }
	T& operator*() { return get(); }
	T* operator->() { return &get(); }
	
	template<typename... Params>
	void attach(Params&&... params) {
		impl -> payload = impl -> payload.attach(fwd<Params>(params)...);
	}
	
	Own<T> asOwn() { return kj::attachRef(get(), *this); }
		
private:
	struct Impl : kj::Refcounted {
		Own<Payload> payload;
		
		template<typename... Params>
		Impl(Params&&... t) :
			payload(kj::heap<Payload>(kj::fwd<Params>(t)...))
		{}
		
		Impl(Own<Payload>&& payload) :
			payload(mv(payload))
		{}
		
		Own<Impl> addRef() {
			return kj::addRef(*this);
		}
	};
	
	mutable Own<Impl> impl;
};

template<typename T>
struct AtomicShared {
	using Payload = T;
	
	template<typename... Params>
	AtomicShared(Params&&... t) :
		impl(kj::atomicRefcounted<Impl>(kj::fwd<Params>(t)...))
	{}
	
	AtomicShared(const AtomicShared<T>& other) :
		impl(other.impl -> addRef())
	{}
	
	AtomicShared(AtomicShared<T>& other) :
		impl(other.impl -> addRef())
	{}
	
	AtomicShared<T>& operator=(const AtomicShared<T>& other) {
		impl = other.impl -> addRef();
	}
	
	AtomicShared(AtomicShared<T>&& other) = default;
	AtomicShared<T>& operator=(AtomicShared<T>&& other) = default;
	
	T& get() { return *(impl -> payload); }
	T& operator*() { return get(); }
	T* operator->() { return &get(); }
	
	// no attach() since we assume Impl to be cross-thread and therefore
	// unsafe to mutate
	
	Own<T> asOwn() { return kj::attachRef(get(), *this); }
	
private:
	struct Impl : kj::AtomicRefcounted {
		mutable Own<Payload> payload;
		
		template<typename... Params>
		Impl(Params&&... t) :
			payload(kj::heap<Payload>(kj::fwd<Params>(t)...))
		{}
		
		Impl(Own<Payload>&& payload) :
			payload(mv(payload))
		{}
		
		Own<const Impl> addRef() const {
			return kj::atomicAddRef(*this);
		}
	};
	
	mutable Own<const Impl> impl;
};

}