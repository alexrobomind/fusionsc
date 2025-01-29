#pragma once

#include "common.h"

namespace fsc {

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
		
	~Shared() noexcept {};
	
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
		
	~AtomicShared() noexcept {};
	
	T& get() { return *(impl -> payload); }
	const T& get() const { return *(impl -> payload); }
	
	T& operator*() { return get(); }
	T* operator->() { return &get(); }
	
	const T& operator*() const { return get(); }
	const T* operator->() const { return &get(); }
	
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