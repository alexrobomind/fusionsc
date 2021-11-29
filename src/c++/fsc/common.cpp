#include "common.h"
#include "data.h"

namespace fsc {
	
Promise<void> removeDatarefsInStruct(capnp::AnyStruct::Reader in, capnp::AnyStruct::Builder out);
Promise<void> removeCapability(capnp::Capability::Client client, capnp::AnyPointer::Builder out);

Promise<void> removeDatarefs(capnp::AnyPointer::Reader in, capnp::AnyPointer::Builder out) {
	using capnp::PointerType;
	using capnp::AnyPointer;
	
	switch(in.getPointerType()) {
		case PointerType::NULL_: {
			out.clear();
		}
		case PointerType::LIST: {
			using capnp::AnyList;
			using capnp::AnyStruct;
			using capnp::ElementSize;
			using capnp::List;
			
			// Retrieve as list
			auto anyList = in.getAs<AnyList>();
			auto inSize = anyList.size();
			
			// Check size format
			switch(anyList.getElementSize()) {
				// Case 1: Pointers
				case ElementSize::POINTER: {
					auto pointerListIn  = anyList.as<List<AnyPointer>>();
					auto pointerListOut = out.initAsAnyList(ElementSize::POINTER, inSize).as<List<AnyPointer>>();
					
					auto promises = kj::heapArrayBuilder<Promise<void>>(inSize);
					for(decltype(inSize) i = 0; i < inSize; ++i)
						promises.add(removeDatarefs(pointerListIn[i], pointerListOut[i]));
					
					return joinPromises(promises.finish());
				}
				
				// Case 2: Structs
				case ElementSize::INLINE_COMPOSITE: {
					// Special case: Size 0
					if(inSize == 0) {
						out.initAsListOfAnyStruct(0, 0, 0);
						return READY_NOW;
					}
					
					auto structListIn  = anyList.as<List<AnyStruct>>();
					auto dataWords = structListIn[0].getDataSection().size() / sizeof(capnp::word);
					auto pointerSize = structListIn[0].getPointerSection().size();
					
					auto structListOut = out.initAsListOfAnyStruct(dataWords, pointerSize, inSize);
					
					auto promises = kj::heapArrayBuilder<Promise<void>>(inSize);
					for(decltype(inSize) i = 0; i < inSize; ++i)
						promises.add(removeDatarefsInStruct(structListIn[i], structListOut[i]));
					
					return joinPromises(promises.finish());
				}
				
				// Other sizes are simple data sections, so we can just copy (phew)
				default:
					out.set(in);
			}
		}
		case PointerType::STRUCT: {
			using capnp::AnyStruct;
			
			auto structIn = in.getAs<AnyStruct>();
			auto dataWords = structIn.getDataSection().size() / sizeof(capnp::word);
			auto pointerSize = structIn.getPointerSection().size();
			return removeDatarefsInStruct(structIn, out.initAsAnyStruct(dataWords, pointerSize));
		}
		case PointerType::CAPABILITY: {
			return removeCapability(in.getAs<capnp::Capability>(), out);
		}
	};
	
	return READY_NOW;
}

Promise<void> removeDatarefsInStruct(capnp::AnyStruct::Reader in, capnp::AnyStruct::Builder out) {
	// Copy over data section
	KJ_ASSERT(out.getDataSection().size() == in.getDataSection().size());
	memcpy(out.getDataSection().begin(), in.getDataSection().begin(), in.getDataSection().size());
	
	// Handle pointers
	auto nPointers = in.getPointerSection().size();
	auto promises = kj::heapArrayBuilder<Promise<void>>(nPointers);
	
	for(decltype(nPointers) i = 0; i < nPointers; ++i) {
		promises.add(removeDatarefs(in.getPointerSection()[i], out.getPointerSection()[i]));
	}
	
	return joinPromises(promises.finish());
}

Promise<void> removeCapability(capnp::Capability::Client client, capnp::AnyPointer::Builder out) {
	using capnp::AnyPointer;
	
	auto typedClient = client.castAs<DataRef<AnyPointer>>();
	
	return typedClient.metadataRequest().send()
	.then([out](auto response) mutable {
		out.setAs<capnp::Data>(response.getMetadata().getId());
	})
	.catch_([out](kj::Exception e) mutable { out.clear(); });
}

}