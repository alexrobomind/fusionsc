#include "load-limiter.h"

#include <kj/async-queue.h>

namespace fsc {

struct LoadLimiter::Impl : public kj::Refcounted {
	Own<Impl> addRef() { return kj::addRef(*this); }
	
	struct TokenImpl : public Token {
		Own<Impl> parent;
		
		TokenImpl(Impl& p) : parent(p.addRef()) { ++p.nActive; }
		~TokenImpl() { --parent -> nActive; parent -> update(); }
	};
	
	size_t capacity;
	size_t nActive = 0;
	size_t nQueued = 0;
	kj::WaiterQueue<Own<Token>> queue;
	
	Impl(size_t cap) : capacity(cap) {}
	
	void update() {
		while(!queue.empty() && nActive < capacity) {
			queue.fulfill(createToken());
			--nQueued;
		}
	}
	
	Own<Token> createToken() {
		return kj::heap<TokenImpl>(*this);
	}
};

LoadLimiter::LoadLimiter(size_t newCap) :
	pImpl(kj::refcounted<Impl>(newCap))
{}

Promise<Own<LoadLimiter::Token>> LoadLimiter::getToken() {
	if(pImpl -> nActive < pImpl -> capacity)
		return pImpl -> createToken();
	
	++pImpl -> nQueued;
	return pImpl -> queue.wait().attach(pImpl -> addRef());
}

size_t LoadLimiter::getCapacity() { return pImpl -> capacity; }

size_t LoadLimiter::getActive() { return pImpl -> nActive; }
size_t LoadLimiter::getQueued() { return pImpl -> nQueued; }

}
