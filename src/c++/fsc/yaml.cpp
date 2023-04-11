#include "yaml.h"

using capnp::DynamicStruct;
using capnp::DynamicList;
using capnp::DynamicValue;

namespace {
	DynamicValue::Reader parsePrimitive(capnp::Type type, YAML::Node src) {
		KJ_REQUIRE(!type.isData());
		KJ_REQUIRE(!type.isText());
		KJ_REQUIRE(!type.isStruct());
		KJ_REQUIRE(!type.isList());
		
		if(type.isEnum()) {
			return capnp::DynamicEnum(type.asEnum().getEnumerantByName(src.as<std::string>()));
		}
		
		if(type.isVoid()) return capnp::Void();		

		if(type.isAnyPointer()) return nullptr;
		if(type.isInterface()) return nullptr;
		
		#define HANDLE_TYPE(checkFun, tp) \
			if(type.checkFun()) return src.as<tp>();
		HANDLE_TYPE(isBool, bool);
		HANDLE_TYPE(isFloat64, double);
		HANDLE_TYPE(isFloat32, float);
		
		HANDLE_TYPE(isInt8, int);
		HANDLE_TYPE(isInt16, int16_t);
		HANDLE_TYPE(isInt32, int32_t);
		HANDLE_TYPE(isInt64, int64_t);
		
		HANDLE_TYPE(isUInt8, unsigned int);
		HANDLE_TYPE(isUInt16, int16_t);
		HANDLE_TYPE(isUInt32, int32_t);
		HANDLE_TYPE(isUInt64, int64_t);
		#undef HANDLE_TYPE
		
		return capnp::Void();
	}
	
	void emitPrimitive(YAML::Emitter& emitter, DynamicValue::Reader val) {
		auto type = val.getType();
				
		KJ_REQUIRE(type != DynamicValue::STRUCT);
		KJ_REQUIRE(type != DynamicValue::LIST);
		
		switch(type) {
			// TODO: DataRefs
			case DynamicValue::CAPABILITY:
				emitter << "<capability>";
				break;
			case DynamicValue::ANY_POINTER:
				emitter << "<unknown>";
				break;
				
			case DynamicValue::VOID:
				emitter << "void";
				break;
				
			case DynamicValue::DATA: {
				auto asBin = val.as<capnp::Data>();
				YAML::Binary data(asBin.begin(), asBin.size());
				emitter << data;
				break;
			}
			
			case DynamicValue::TEXT:
			case DynamicValue::ENUM:
				emitter << val.as<capnp::Text>().cStr();
				break;
			
			case DynamicValue::BOOL:
				emitter << val.as<bool>();
				break;
			
			case DynamicValue::FLOAT:
				emitter << val.as<double>();
				break;
			
			case DynamicValue::INT:
				emitter << val.as<int64_t>();
				break;
			
			case DynamicValue::UINT:
				emitter << val.as<uint64_t>();
				break;
			
			default:
				KJ_FAIL_REQUIRE("Internal error: Unhandled type");
		}
	}
}

namespace fsc {
	
void load(DynamicList::Builder dst, YAML::Node src) {
	auto listSchema = dst.getSchema();
	auto elType = listSchema.getElementType();
	
	for(auto i : kj::indices(dst)) {
		YAML::Node val = src[i];
		
		if(elType.isStruct()) {
			load(dst[i].as<DynamicStruct>(), val);
		} else if(elType.isList()) {
			auto out = dst.init(i, val.size());
			load(out.as<DynamicList>(), val);
		} else if(elType.isData()) {
			auto asBinary = src.as<YAML::Binary>();
			capnp::Data::Reader dataPtr(asBinary.data(), asBinary.size());
			dst.set(i, dataPtr);
		} else if(elType.isText()) {
			auto asString = src.as<std::string>();
			dst.set(i, capnp::Text::Reader(asString));
		} else {
			dst.set(i, parsePrimitive(elType, val));
		}
	}
}

void load(DynamicStruct::Builder dst, YAML::Node src) {
	// Interpret nodes as map
	for(auto it = src.begin(); it != src.end(); ++it) {
		// Extract key and value
		std::string key = it -> first.as<std::string>();
		YAML::Node val = it -> second;
		
		Maybe<kj::String> previouslySetUnionField = nullptr;
		
		// Check if we can match the field name
		KJ_IF_MAYBE(pField, dst.getSchema().findFieldByName(key)) {
			capnp::StructSchema::Field& field = *pField;
			
			// Make sure we don't set union fields twice
			if(field.getProto().getDiscriminantValue() != capnp::schema::Field::NO_DISCRIMINANT) {
				KJ_IF_MAYBE(pPrevSet, previouslySetUnionField) {
					kj::StringPtr currentField = key;
					kj::StringPtr previousField = *pPrevSet;
					
					KJ_FAIL_REQUIRE("Can not set two fields of the same union", currentField, previousField);
				}
				previouslySetUnionField = kj::str(key);
			}
			
			capnp::Type type = field.getType();
			
			auto elType = type;
			while(elType.isList()) {
				elType = elType.asList().getElementType();
			}
			
			KJ_REQUIRE(!elType.isAnyPointer(), "Can not specify AnyPointer-typed fields with YAML");
			KJ_REQUIRE(!elType.isInterface(), "Interface capabilities can not be set via YAML");
			
			if(type.isStruct()) {
				auto fieldVal = dst.init(field).as<capnp::DynamicStruct>();
				load(fieldVal, val);
				return;
			}
			if(type.isList()) {
				auto fieldVal = dst.init(field, val.size()).as<capnp::DynamicList>();
				load(fieldVal, val);
				return;
			}
			if(type.isData()) {
				auto asBinary = val.as<YAML::Binary>();
				capnp::Data::Reader dataPtr(asBinary.data(), asBinary.size());
				dst.set(field, dataPtr);
				return;
			}
			if(type.isText()) {				
				auto asString = val.as<std::string>();
				dst.set(field, capnp::Text::Reader(asString));
				return;
			}
			
			dst.set(field, parsePrimitive(type, val));
		} else {
			KJ_LOG(WARNING, "Could not find matching field for map entry", key);
			// KJ_DBG(dst.getSchema(), kj::StringPtr(key));
		}
	}
}

YAML::Emitter& operator<<(YAML::Emitter& dst, DynamicList::Reader src) {
	auto listSchema = src.getSchema();
	auto elType = listSchema.getElementType();
	
	if(!elType.isStruct() && !elType.isList() && !elType.isData() && !elType.isText())
		dst << YAML::Flow;
	
	dst << YAML::BeginSeq;
	
	for(auto el : src) {
		if(elType.isStruct()) {
			dst << el.as<DynamicStruct>();
		} else if(elType.isList()) {
			dst << el.as<DynamicList>();
		} else {
			emitPrimitive(dst, el);
		}
	}
	
	dst << YAML::EndSeq;
	
	return dst;
}
	
YAML::Emitter& operator<<(YAML::Emitter& dst, DynamicStruct::Reader src) {
	dst << YAML::BeginMap;
	
	auto emitField = [&](capnp::StructSchema::Field field) {
		auto type = field.getType();
		
		// Inspect inner-most type for lists
		auto elType = type;
		while(elType.isList()) {
			auto asList = elType.asList();
			elType = asList.getElementType();
		}
		
		// Don't emit interface- or any-typed fields or nested lists
		if(elType.isInterface() || elType.isAnyPointer())
			return;
		
		auto val = src.get(field);
		
		dst << YAML::Key << field.getProto().getName();
		dst << YAML::Value;
		
		if(type.isStruct()) {
			dst << val.as<DynamicStruct>();
		} else if(type.isList()) {
			dst << val.as<DynamicList>();
		} else {
			emitPrimitive(dst, val);
		}
	};
	
	auto structSchema = src.getSchema();
	for(auto field : structSchema.getNonUnionFields()) {
		emitField(field);
	}
	
	KJ_IF_MAYBE(pField, src.which()) {
		emitField(*pField);
	}	
			
	dst << YAML::EndMap;
	
	return dst;
}

}