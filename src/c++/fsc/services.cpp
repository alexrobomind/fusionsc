#include "services.h"
#include "magnetics.h"
#include "kernels.h"

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
	
private:
	LibraryThread lt;
};

struct ResolverChainImpl : public virtual capnp::Capability::Server, public virtual capnp::ResolverChain::Server {
	using capnp::Capability::Server::DispatchCallResult;
	
	struct Registration {
		ResolverChainImpl& parent;
		ListLink<Registration> link;
		
		capnp::Capability::Client entry;

		Registration(ResolverChainImpl& parent, capnp::Capability::Client entry);
		~Registration();
	}
	
	kj::List<Registration, &Registration::link> registrations;
	
	Promise<void> register(RegisterContext ctx) override {
		auto result = capnp::newBrokenCap(KJ_EXCEPTION(UNIMPLEMENTED, "Unimplemented"));
		
		result = attach(result, 
			thisCap(),
			kj::heap<Registration>(*this, ctx.getParams().getResolver())
		);
	};
	
	static void rethrowIfNotUnimplemented(kj::Exception&& exc) {
		
	}
	
	DispatchCallResult dispatchCall(uint64_t interfaceId, uint16_t methodId, CallContext<AnyPointer, AnyPointer> ctx) {
		Promise<void> result = mv(capnp::ResolverChain::Server::dispatchCall(interfaceId, methodId, ctx).promise)
		.catch_([=, this](kj::Exception&& exc) {
			using capnp::AnyStruct;
			using capnp::AnyPointer;
			
			// We only handle generic methods if parent didnt have it implemented
			if(exc.getType() != kj::Exception::Type::UNIMPLEMENTED) {
				kj::throwRecoverableException(exc);
			}
			
			// Updatable container on heap
			auto paramMsg = kj::heap<Own<MallocMessageBuilder>>();
			
			// Fill initial value from context
			*paramMsg = kj::heap<MallocMessageBuilder>();
			*paramMsg->setRoot(ctx.getParams());
			ctx.releaseParams();
			
			// Check that first field is set
			auto paramsStruct = (**paramMsg).getRoot<AnyStruct>();
			auto pSec = paramsStruct.getPointerSection();
			KJ_REQUIRE(pSec.size() > 0);
			
			Promise<void> result = READY_NOW;
			for(auto& reg : registrations) {
				result = result.then([=, this, &paramMsg = *paramMsg]() {
					auto paramsStruct = paramMsg->getRoot<AnyStruct>();
					auto pSec = paramsStruct.getPointerSection();
					
					auto request = reg.entry.typelessRequest(
						interfaceId, methodId, paramsStruct.messageSize()
					);
					request.getParams().set(paramsStruct);
					
					return request.send();
				}).then([=, this, &paramMsg = *paramMsg](auto result) {
					// Copy old extra parameters into new message, but drop old result
					{
						auto paramsStruct = paramMsg->getRoot<AnyStruct>();
						auto pSec = paramsStruct.getPointerSection();		
						pSec[0].clear();
						
						auto newParamMsg = kj::heap<MallocMessageBuilder>();
						newParamMsg->setRoot(paramsStruct);
						paramMsg = mv(newParamMsg);
					}
					
					// Copy result into params
					auto paramsStruct = paramMsg->getRoot<AnyStruct>();
					auto pSec = paramsStruct.getPointerSection();		
					pSec[0].setAs(result);
				}).catch_([]kj::Exception&& e) {
				});
			}
			
			result = result.then([=, this, &paramMsg = *paramMsg]() {
				ctx.getResults().setAs(paramMsg->getRoot<AnyPointer>());
			});
			
			return result.attach(mv(paramMsg));
		}
		
		return { mv(result), false };
	}
	
	Promise<void> resolveField(ResolveFieldContext ctx) {
		Promise<void> result = READY_NOW;
		
		auto pField = kj::heap<Own<Temporary<MagneticField>>>();
		*pField = kj::heap<Temporary<MagneticField>>(ctx.getParams().getField());
		
		for(auto& reg : registrations) {
			result = result.then([ctx, &field = *pField]() {
				auto request = reg.entry.castAs<FieldResolver>().resolveFieldRequest();
				auto params  = request.getParams();
				params.setField(field->asBuilder());
				params.setFollowRefs(ctx.getParams().getFollowRefs());
				
				return request.send();
			}).then([&field = *pField](auto result) {
				field = kj::heap<Temporary<MagneticField>>(result.getField);
			}).catch_([](kj::Exception& e) {
				if(e.getType() == kj::Exception::Type::UNIMPLEMENTED);
		}
	}
};

ResolverChainImpl::Registration::Registration(ResolverChainImpl& parent, capnp::Capability::Client entry) :
	parent(parent), entry(entry)
{
	parent.registrations.add(*this);
}

ResolverChainImpl::Registration::~Registration() {
	parent.registrations.remote(*this);
}

}

RootService::Client fsc::createRoot(LibraryThread& lt, RootConfig::Reader config) {
	return kj::heap<RootServer>(lt, config);
}