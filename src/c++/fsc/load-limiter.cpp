#include "load-limiter.h"

#include <kj/async-queue.h>

namespace fsc {

struct LoadLimiter::Impl : public kj::Refcounted {
	Own<Impl> addRef() { return kj::addRef(this); }
	
	struct TokenImpl {
		Own<Impl> parent;
		
		TokenImpl(Own<Impl> parent) : parent(mv(parent)) { ++parent -> nActive; }
		~TokenImpl() { --parent -> nActive; parent -> update(); }
	};
	
	size_t capacity;
	size_t nActive = 0;
	kj::WaiterQueue<Token> queue;
	
	Impl(size_t cap) : capacity(cap) {}
	
	void update() {
		while(!queue.empty() && nActive < capacity) {
			queue.fulfill(createToken());
		}
	}
	
	Token createToken() {
		return Token { kj::heap<TokenImpl>(addRef()) };
	}
};

LoadLimiter::LoadLimiter(size_t newCap) :
	pImpl(kj::refcounted<Impl>(newCap))
{}

Promise<LoadLimiter::Token> LoadLimiter::getToken() {
	if(nActive < capacity)
		return createToken();
	
	return queue.wait();
}

}