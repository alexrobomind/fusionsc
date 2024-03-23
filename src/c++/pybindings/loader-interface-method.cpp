#include "fscpy.h"
#include "assign.h"
#include "async.h"

using kj::str;

using capnp::RemotePromise;
using capnp::Response;
using capnp::DynamicCapability;
using capnp::DynamicStruct;
using capnp::DynamicValue;
using capnp::StructSchema;
using capnp::InterfaceSchema;
using capnp::AnyPointer;

namespace fscpy {

namespace {


struct TypeInference {	
	kj::Vector<Maybe<capnp::Type>> paramBindings;
	kj::String errorMsg = str("");
	
	capnp::SchemaLoader& loader;
	
	TypeInference(capnp::SchemaLoader& l) : loader(l) {}
	
	void clear() { paramBindings.clear(); errorMsg = str(""); }
	
	bool infer(capnp::Type from, capnp::Type to, bool strict) {
		if(to.isStruct()) {
			if(!from.isStruct()) {
				errorMsg = str("Attempting to use non-struct type as struct ", to.asStruct());
				return false;
			}
			
			return inferSchema(from.asStruct(), to.asStruct());
		}
		
		if(to.isInterface()) {
			if(!from.isInterface()) {
				errorMsg = str("Attempting to use non-interface type as interface ", to.asInterface());
				return false;
			}
			
			if(strict) {
				return inferSchema(from.asInterface(), to.asInterface());
			}
			
			auto superclass = from.asInterface().findSuperclass(to.asInterface().getProto().getId());
			KJ_IF_MAYBE(pSc, superclass) {
				return inferSchema(*pSc, to.asInterface());
			} else {
				errorMsg = str("Interface ", from.asInterface(), " does not extend ", to.asInterface());
				return false;
			}
		}
		
		if(to.isList()) {
			if(!from.isList()) {
				errorMsg = str("Attempting to use non-list as list");
				return false;
			}
			
			return infer(from.asList().getElementType(), to.asList().getElementType(), strict);
		}
		
		if(to.which() == capnp::schema::Type::ANY_POINTER) {
			if(!from.isInterface() && !from.isList() && !from.isStruct() && !from.isAnyPointer()) {
				errorMsg = str("Only structs, interfaces, and lists can be used for AnyPointer parameters (incl. type parameters)");
				return false;
			}
			
			KJ_IF_MAYBE(pImplicit, to.getImplicitParameter()) {	
				unsigned int idx = pImplicit -> index;
				
				if(idx >= paramBindings.size()) {
					paramBindings.resize(idx + 1);
				}
				
				KJ_IF_MAYBE(pBinding, paramBindings[idx]) {
					return infer(from, *pBinding, true);
				} else {
					paramBindings[idx] = from;
					return true;
				}
			}
		}
		
		return true;
	}
	
	capnp::Type specialize(capnp::Type type) {		
		Temporary<capnp::schema::Type> proto;
		extractType(type, proto);
		
		specialize(proto);
		
		return loader.getType(proto.asReader());
	}

private:
	bool inferSchema(capnp::Schema t1, capnp::Schema t2) {
		if(t2.getGeneric() != t1.getGeneric()) {
			errorMsg = str("Can not use ", t1, " as ", t2);
			return false;
		}
		
		for(auto scopeId : t2.getGenericScopeIds()) {
			auto brands1 = t1.getBrandArgumentsAtScope(scopeId);
			auto brands2 = t2.getBrandArgumentsAtScope(scopeId);
			
			for(auto iBrand : kj::range(0, kj::min(brands1.size(), brands2.size()))) {
				if(!infer(brands1[iBrand], brands2[iBrand], true)) return false;
			}
		}
		
		return true;
	}
	
	void specialize(capnp::schema::Type::Builder x) {
		if(x.isList()) { specialize(x.getList().getElementType()); }
		if(x.isEnum()) { specialize(x.getEnum().getBrand()); }
		if(x.isStruct()) { specialize(x.getStruct().getBrand()); }
		if(x.isInterface()) { specialize(x.getInterface().getBrand()); }
		
		if(x.isAnyPointer()) {
			auto aptr = x.getAnyPointer();
			if(aptr.isImplicitMethodParameter()) {
				auto idx = aptr.getImplicitMethodParameter().getParameterIndex();
				
				Maybe<capnp::Type> target = nullptr;
				if(idx < paramBindings.size())
					target = paramBindings[idx];
				
				KJ_IF_MAYBE(pTarget, target) {
					Temporary<capnp::schema::Type> asProto;
					extractType(*pTarget, asProto);
					
					// Copy over target onto x
					auto anyIn = capnp::toAny(asProto.asReader());
					auto anyOut = capnp::toAny(x);
					
					auto dataIn = anyIn.getDataSection();
					auto dataOut = anyOut.getDataSection();
					memcpy(dataOut.begin(), dataIn.begin(), kj::min(dataOut.size(), dataIn.size()));
					
					if(dataOut.size() > dataIn.size()) {
						memset(dataOut.begin() + dataIn.size(), 0, dataOut.size() - dataIn.size());
					}
					
					auto ptrIn = anyIn.getPointerSection();
					auto ptrOut = anyOut.getPointerSection();
					
					for(auto i : kj::indices(ptrOut)) {
						if(i < ptrIn.size()) {
							ptrOut[i].setAs<capnp::AnyPointer>(ptrIn[i]);
						} else {
							ptrOut[i].clear();
						}
					}
				} else {
					KJ_LOG(WARNING, "A generic method type could not be deduced. This can happen if the relevant parameter is passed in as python types (dict, etc.). The following errors appeared during type deduction: ", errorMsg);
					aptr.initUnconstrained().setAnyKind();
				}
			}
		}
	}
	
	void specialize(capnp::schema::Brand::Builder x) {
		for(auto scope : x.getScopes()) {
			if(!scope.isBind())
				continue;
			
			for(auto binding : scope.getBind()) {
				if(binding.isType()) {					
					specialize(binding.getType());
				}
			}
		}
	}
};


//! Handles the exposure of Cap'n'proto interface methods as python methods
struct InterfaceMethod {
	capnp::InterfaceSchema::Method method;
	
	capnp::StructSchema paramType;
	capnp::StructSchema resultType;
	
	kj::Vector<capnp::StructSchema::Field> argFields;
	
	capnp::SchemaLoader& loader;
	
	kj::String name;
	
	InterfaceMethod(capnp::InterfaceSchema::Method method, capnp::SchemaLoader& loader) :
		method(method), loader(loader)
	{
		kj::StringPtr rawName = method.getProto().getName();
		name = memberName(rawName);
				
		auto makeDefaultBrand = [&loader](uint64_t nodeId, capnp::schema::Brand::Builder out) {
			// Creates a default brand that binds each argument to a method parameter
			auto structInfo = loader.get(nodeId);
			auto proto = structInfo.getProto();
			
			if(proto.getParameters().size() == 0)
				return;
			
			auto scope = out.initScopes(1)[0];
			scope.setScopeId(nodeId);
			
			auto bindings = scope.initBind(proto.getParameters().size());
			for(auto iBinding : kj::indices(bindings)) {
				auto binding = bindings[iBinding];
				binding.initType().initAnyPointer().initImplicitMethodParameter().setParameterIndex(iBinding);
			}
		};
		
		Temporary<capnp::schema::Brand> paramBrand = method.getProto().getParamBrand();
		Temporary<capnp::schema::Brand> resultBrand = method.getProto().getResultBrand();
		
		if(paramBrand.getScopes().size() == 0) {
			makeDefaultBrand(method.getProto().getParamStructType(), paramBrand);
		}
		
		if(resultBrand.getScopes().size() == 0) {
			makeDefaultBrand(method.getProto().getResultStructType(), resultBrand);
		}
		
		paramType = loader.get(method.getProto().getParamStructType(), paramBrand.asReader()).asStruct();
		resultType = loader.get(method.getProto().getResultStructType(), resultBrand.asReader()).asStruct();	
		
		auto doc = kj::strTree();
		
		size_t nArgs = 0;
				
		for(auto field : paramType.getNonUnionFields()) {
			// Only process slot fields
			if(field.getProto().which() != capnp::schema::Field::SLOT)
				continue;
			
			argFields.add(field);
		}
	}
	
	py::object operator()(capnp::DynamicCapability::Client self, py::args pyArgs, py::kwargs pyKwargs) const {	
		// auto request = self.newRequest(method);
		capnp::Request<AnyPointer, AnyPointer> request = self.typelessRequest(
			method.getContainingInterface().getProto().getId(), method.getOrdinal(), nullptr, capnp::Capability::Client::CallHints()
		);
		
		// Check whether we got the argument structure passed
		// In this case, copy the fields over from input struct
		
		TypeInference typeInference(loader);
		
		bool requestBuilt = false;
		
		if(py::len(pyKwargs) == 0 && py::len(pyArgs) == 1) {
			// Check whether the first argument has the correct type
			auto argVal = pyArgs[0].cast<DynamicValueReader>();
			
			if(argVal.getType() == DynamicValue::STRUCT && argVal.asStruct().getSchema().getProto().getId() == paramType.getProto().getId()) {
				DynamicStruct::Reader asStruct = argVal.as<DynamicStruct>();
				
				if(!typeInference.infer(asStruct.getSchema(), paramType, false)) {
					KJ_FAIL_REQUIRE("Failed to match parameter type against target type", typeInference.errorMsg);
				}
				
				request.setAs<DynamicStruct>(asStruct);
				requestBuilt = true;
			}
		}
			

		if(!requestBuilt) {
			// Parse positional arguments
			KJ_REQUIRE(pyArgs.size() <= argFields.size(), "Too many arguments specified");
			
			bool inferTypes = paramType.getProto().getParameters().size() > 0;
			
			auto nameList = py::list(pyKwargs);
			
			if(inferTypes) {
				auto inferType = [&typeInference](capnp::Type dst, DynamicValueReader reader) {
					Maybe<capnp::Type> asType;
					if(dst.isStruct()) {
						KJ_REQUIRE(reader.getType() == capnp::DynamicValue::STRUCT);	
						asType = reader.asStruct().getSchema();
					} else if(dst.isInterface()) {
						KJ_REQUIRE(reader.getType() == capnp::DynamicValue::CAPABILITY);
						asType = reader.as<capnp::DynamicCapability>().getSchema();
					} else if(dst.isList()) {
						KJ_REQUIRE(reader.getType() == capnp::DynamicValue::LIST);
						asType = reader.asList().getSchema();
					}
					
					KJ_IF_MAYBE(pType, asType) {
						KJ_REQUIRE(typeInference.infer(*pType, dst, false), "Failed to match parameter type", typeInference.errorMsg);
					}
				};
				
				// Try to derive proper type for fields
				for(size_t i = 0; i < pyArgs.size(); ++i) {
					auto field = argFields[i];
										
					py::detail::type_caster<DynamicValueReader> readerCaster;
					KJ_REQUIRE(readerCaster.load(pyArgs[i], false), "Failed to convert positional argument", i);
					
					inferType(field.getType(), readerCaster.operator DynamicValueReader&());
				}
				
				auto inferEntry = [this, &inferType](kj::StringPtr name, DynamicValueReader value) mutable {
					KJ_IF_MAYBE(pField, paramType.findFieldByName(name)) {
						inferType(pField -> getType(), value);
					} else {
						KJ_FAIL_REQUIRE("Unknown named parameter", name);
					}
				};
				auto pyInferEntry = py::cpp_function(inferEntry);
				
				for(auto name : nameList) {
					pyInferEntry(name, pyKwargs[name]);
				}
			}
			
			// Specialize types
			auto specializedParamType = typeInference.specialize(paramType).asStruct();
			DynamicStruct::Builder structRequest = request.initAs<DynamicStruct>(specializedParamType);
			
			for(size_t i = 0; i < pyArgs.size(); ++i) {
				auto field = argFields[i];
				assign(structRequest, field.getProto().getName(), pyArgs[i]);
			}
			
			// Parse keyword arguments
			for(auto name : nameList) {
				py::detail::make_caster<kj::StringPtr> nameCaster;
				KJ_REQUIRE(nameCaster.load(name, false), "Could not convert kwarg name to C++ string");
				kj::StringPtr kjName = nameCaster.operator kj::StringPtr&();
				
				KJ_IF_MAYBE(pField, specializedParamType.findFieldByName(kjName)) {
					assign(structRequest, kjName, pyKwargs[name]);
				} else {
					KJ_FAIL_REQUIRE("Unknown named parameter", kjName);
				}
			}
		}
		
		RemotePromise<AnyPointer> result = request.send();
		
		auto specializedResultType = typeInference.specialize(resultType).asStruct();
		
		// Extract promise
		Promise<py::object> resultPromise = result.then([specializedResultType](capnp::Response<AnyPointer> response) {
			DynamicStruct::Reader asStruct = response.getAs<DynamicStruct>(specializedResultType);
			return py::cast(DynamicStructReader(kj::heap(mv(response)), asStruct));
		});
		
		// Extract pipeline
		AnyPointer::Pipeline resultPipelineTypeless = mv(result);
		DynamicStructPipeline resultPipeline(mv(resultPipelineTypeless), specializedResultType);
		
		return convertCallPromise(mv(resultPromise), mv(resultPipeline));
	}
	
	kj::StringTree description() {
		auto isGeneratedStruct = [](capnp::Type type) {
			if(!type.isStruct())
				return false;
			
			auto asStruct = type.asStruct();
			auto displayName = asStruct.getProto().getDisplayName();
			
			KJ_IF_MAYBE(dontCare, displayName.findFirst('$')) {
				return true;
			} else {
				return false;
			}
		};
	
		kj::Vector<kj::StringTree> argumentDescs;
		
		for(auto field : argFields) {
			KJ_REQUIRE(field.getProto().isSlot());
			auto slot = field.getProto().getSlot();
			
			auto name = field.getProto().getName();
			auto type = field.getType();
			
			argumentDescs.add(strTree(typeName(type), " ", name));
		}
		
		auto paramName = isGeneratedStruct(paramType) ? strTree(name, ".", typeName(paramType)) : typeName(paramType);
		auto resultName = isGeneratedStruct(resultType) ? strTree(name, ".", typeName(resultType)) : typeName(resultType);
		
		auto functionDesc = kj::strTree(
			name, " : (", kj::StringTree(argumentDescs.releaseAsArray(), ", "), ") -> ", resultName.flatten(), ".Promise\n",
			"\n",
			"    or alternatively \n",
			"\n",
			name, " : (", paramName.flatten(), ") -> ", resultName.flatten(), ".Promise"
		);
		
		return functionDesc;
	}
};

}

py::object Loader::makeInterfaceMethod(capnp::InterfaceSchema::Method method) {
	InterfaceMethod backend(method, this -> capnpLoader);
	kj::StringTree desc = backend.description();
	kj::String name = kj::heapString(backend.name);
			
	auto pyFunction = py::cpp_function(
		mv(backend),
		py::return_value_policy::move,
		py::doc(desc.flatten().cStr()),
		py::arg("self")
	);
	
	kj::StringPtr moduleName = kj::get<0>(this -> qualName(method));
		
	py::object descriptor = methodDescriptor(pyFunction);
	descriptor.attr("__name__") = name.cStr();
	descriptor.attr("__module__") = moduleName.cStr();
	
	return descriptor;
}

}