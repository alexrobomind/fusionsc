#include "services.h"
#include "magnetics.h"
#include "kernels.h"
#include "flt.h"

#include <kj/list.h>

using namespace fsc;

namespace {
	
template<typename T>
auto selectDevice(T t, WorkerType preferredType) {
	#ifdef FSC_WITH_CUDA
	
	try {
		if(preferredType == WorkerType::GPU) {
			return tuple(t(newGpuDevice()), WorkerType::GPU);
		}
	} catch(kj::Exception& e) {
	}
	
	#endif
	
	return tuple(t(newThreadPoolDevice()), WorkerType::CPU);
}

struct RootServer : public RootService::Server {
	RootServer(LibraryThread& lt, RootConfig::Reader config) :
		lt(lt->addRef())
	{}
	
	Promise<void> newFieldCalculator(NewFieldCalculatorContext context) {
		auto factory = [this, context](auto device) mutable {			
			return ::fsc::newFieldCalculator(lt, context.getParams().getGrid(), mv(device));
		};
		
		auto selectResult = selectDevice(factory, context.getParams().getPreferredDeviceType());
		
		auto results = context.getResults();
		results.setCalculator(kj::get<0>(selectResult));
		results.setDeviceType(kj::get<1>(selectResult));
		
		return READY_NOW;
	}
	
	Promise<void> newTracer(NewTracerContext context) {
		context.initResults().setService(newFLT(lt, newThreadPoolDevice()));
		return READY_NOW;
	}
	
private:
	LibraryThread lt;
};

struct ResolverChainImpl : public virtual capnp::Capability::Server, public virtual ResolverChain::Server {
	using capnp::Capability::Server::DispatchCallResult;
	using ResolverChain::Server::RegisterContext;
	
	struct Registration {
		ResolverChainImpl& parent;
		kj::ListLink<Registration> link;
		
		capnp::Capability::Client entry;

		Registration(ResolverChainImpl& parent, capnp::Capability::Client entry);
		~Registration();
	};
	
	kj::List<Registration, &Registration::link> registrations;
	
	Promise<void> register_(RegisterContext ctx) override {		
		auto result = attach(
			ResolverChain::Client(capnp::newBrokenCap(KJ_EXCEPTION(UNIMPLEMENTED, "Unimplemented"))),
			
			thisCap(),
			kj::heap<Registration>(*this, ctx.getParams().getResolver())
		);
		ctx.initResults().setRegistration(mv(result));
		
		return READY_NOW;
	};
	
	DispatchCallResult dispatchCall(uint64_t interfaceId, uint16_t methodId, capnp::CallContext<capnp::AnyPointer, capnp::AnyPointer> ctx) override {
		Promise<void> result = ResolverChain::Server::dispatchCall(interfaceId, methodId, ctx).promise
		.catch_([=](kj::Exception&& exc) mutable -> Promise<void> {
			using capnp::AnyStruct;
			using capnp::AnyPointer;
			
			// We only handle generic methods if parent didnt have it implemented
			if(exc.getType() != kj::Exception::Type::UNIMPLEMENTED) {
				kj::throwRecoverableException(mv(exc));
				return READY_NOW;
			}
			
			// Updatable container on heap
			auto paramMsg = heapHeld<Own<capnp::MallocMessageBuilder>>();
			
			// Fill initial value from context
			*paramMsg = kj::heap<capnp::MallocMessageBuilder>();
			(**paramMsg).setRoot(ctx.getParams());
			ctx.releaseParams();
			
			// Check that first field is set
			{
				auto paramsStruct = (**paramMsg).getRoot<AnyStruct>();
				auto pSec = paramsStruct.getPointerSection();
				KJ_REQUIRE(pSec.size() > 0);
			}
			
			Promise<void> result = READY_NOW;
			for(auto& reg : registrations) {
				result = result.then([=, e = cp(reg.entry)]() mutable {
					auto params = (**paramMsg).getRoot<AnyPointer>();
					
					auto request = e.typelessRequest(
						interfaceId, methodId, params.targetSize()
					);
					request.set(params);
					
					return request.send();
				}).then([=](auto result) mutable {
					// Copy old extra parameters into new message, but drop old result
					{
						auto paramsStruct = (**paramMsg).getRoot<AnyStruct>();
						auto params       = (**paramMsg).getRoot<AnyPointer>();
						
						auto pSec = paramsStruct.getPointerSection();		
						pSec[0].clear();
						
						auto newParamMsg = kj::heap<capnp::MallocMessageBuilder>();
						newParamMsg->setRoot(params.asReader());
						*paramMsg = mv(newParamMsg);
					}
					
					// Copy result into params
					auto paramsStruct = (**paramMsg).getRoot<AnyStruct>();
					auto pSec = paramsStruct.getPointerSection();		
					pSec[0].set(result);
				}).catch_([](kj::Exception&& e) mutable {
					KJ_LOG(WARNING, "Exception in resolver chain", mv(e));
				});
			}
			
			result = result.then([=]() mutable {
				auto paramsStruct = (**paramMsg).getRoot<AnyStruct>();
				auto pSec = paramsStruct.getPointerSection();
				
				ctx.getResults().set(pSec[0]);
			});
			
			return result.attach(paramMsg.x(), thisCap());
		});
		
		return { mv(result), false };
	}
};

ResolverChainImpl::Registration::Registration(ResolverChainImpl& parent, capnp::Capability::Client entry) :
	parent(parent), entry(entry)
{
	parent.registrations.add(*this);
}

ResolverChainImpl::Registration::~Registration() {
	parent.registrations.remove(*this);
}

}

RootService::Client fsc::createRoot(LibraryThread& lt, RootConfig::Reader config) {
	return kj::heap<RootServer>(lt, config);
}

ResolverChain::Client fsc::newResolverChain() {
	return kj::heap<ResolverChainImpl>();
}