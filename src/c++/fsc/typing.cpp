#include "typing.h"
#include "common.h"

void fsc::extractType(capnp::Type in, capnp::schema::Type::Builder out) {	
	switch(in.which()) {
		#define HANDLE_VALUE(enumVal, name) \
			case capnp::schema::Type::enumVal: {\
				auto outTyped = out.init ## name(); \
				auto inTyped  = in.as ## name(); \
				outTyped.setTypeId(inTyped.getProto().getId()); \
				extractBrand(inTyped, outTyped.initBrand()); \
				break; \
			}
			
		HANDLE_VALUE(ENUM, Enum);
		HANDLE_VALUE(INTERFACE, Interface);
		HANDLE_VALUE(STRUCT, Struct);
		
		#undef HANDLE_VALUE
		
		case capnp::schema::Type::LIST: {
			auto outAsList = out.initList();
			extractType(in.asList().getElementType(), outAsList.getElementType());
			break;
		}
		case capnp::schema::Type::ANY_POINTER: {
			auto outAsAny = out.initAnyPointer();
			
			KJ_IF_MAYBE(pBrandParameter, in.getBrandParameter()) {
				auto outAsParam = outAsAny.initParameter();
				outAsParam.setScopeId(pBrandParameter->scopeId);
				outAsParam.setParameterIndex(pBrandParameter->index);
				break;
			}
			
			KJ_IF_MAYBE(pImplicitParameter, in.getImplicitParameter()) {
				auto outAsParam = outAsAny.initImplicitMethodParameter();
				outAsParam.setParameterIndex(pImplicitParameter->index);
				break;
			}
			
			auto outAsUnconstrained = outAsAny.initUnconstrained();
			switch(in.whichAnyPointerKind()) {
				#define HANDLE_VALUE(enumVal, name) \
					case capnp::schema::Type::AnyPointer::Unconstrained::enumVal: { \
						outAsUnconstrained.set ## name(); \
						break; \
					}
				
				HANDLE_VALUE(ANY_KIND, AnyKind);
				HANDLE_VALUE(STRUCT, Struct);
				HANDLE_VALUE(LIST, List);
				HANDLE_VALUE(CAPABILITY, Capability);
				
				#undef HANDLE_VALUE
			}
			
			break;
		}
		
		#define HANDLE_VALUE(enumVal, name) \
		case capnp::schema::Type::enumVal: \
			out.set ## name(); \
			break;
			
		HANDLE_VALUE(VOID, Void);
		HANDLE_VALUE(BOOL, Bool);
		
		HANDLE_VALUE(INT8, Int8);
		HANDLE_VALUE(INT16, Int16);
		HANDLE_VALUE(INT32, Int32);
		HANDLE_VALUE(INT64, Int64);
		
		HANDLE_VALUE(UINT8, Uint8);
		HANDLE_VALUE(UINT16, Uint16);
		HANDLE_VALUE(UINT32, Uint32);
		HANDLE_VALUE(UINT64, Uint64);
		
		HANDLE_VALUE(FLOAT32, Float32);
		HANDLE_VALUE(FLOAT64, Float64);
		
		HANDLE_VALUE(TEXT, Text);
		HANDLE_VALUE(DATA, Data);
		
		#undef HANDLE_VALUE
	}
}

void fsc::extractBrand(capnp::Schema in, capnp::schema::Brand::Builder out) {
	if(!in.getProto().getIsGeneric()) {
		out.initScopes(0);
		return;
	}
	
	auto scopeIds = in.getGenericScopeIds();
	
	auto outScopes = out.initScopes(scopeIds.size());
	for(auto iScope : kj::indices(scopeIds)) {
		auto outScope = outScopes[iScope];
		outScope.setScopeId(scopeIds[iScope]);
		
		auto inBindings  = in.getBrandArgumentsAtScope(scopeIds[iScope]);
		auto outBindings = outScope.initBind(inBindings.size());
		
		for(auto iBinding : kj::indices(outBindings)) {
			capnp::Type inType = inBindings[iBinding];
			auto outBinding = outBindings[iBinding];
			
			extractType(inType, outBinding.initType());
		}
	}
}