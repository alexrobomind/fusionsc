#pragma once

#include "common.h"

namespace fsc {
	
struct LoadLimiter {
	class Token : public Own<void> { using Own<void>::Own; };
	
	LoadLimiter(size_t capacity = 1);
	
	size_t getCapacity();
	void setCapacity(size_t newCapacity);
	
	size_t getActive();
	
	Promise<Token> getToken();
	
	template<typename C, typename T = capnp::FromClient<C>>
	typename T::Client limit(C);
	
	struct Impl;
	
private:
	Own<Impl> pImpl;
};

KJ_DECLARE_NON_POLYMORPHIC(LoadLimiter::Impl);

// Inline implementation

template<typename C, typename T>
typename T::Client LoadLimiter::limit(C client) {
	return getToken()
	.then([c = mv(client)](Token t) mutable -> typename T::Client {
		return attach(mv(c), mv(t));
	});
}

}