#pragma once

#include "common.h"

namespace fsc {
	
struct LoadLimiter {
	class Token { Own<void> pImpl; };
	
	LoadLimiter(size_t capacity = 1);
	
	void setCapacity(size_t newCapacity);
	Promise<Token> getToken();
	
	template<typename C, typename T = capnp::FromClient<C>>
	typename T::Client limit(C);
	
private:
	struct Impl;
	Own<Impl> pImpl;
};

// Inline implementation

template<typename C, typename T = capnp::FromClient<C>>
typename T::Client LoadBalancer::limit(C client) {
	return getToken()
	.then([c = mv(client)](Token t) {
		return attach(mv(c), mv(t));
	});
}

}