#include "yaml.h"

namespace {
	DynamicValue::Reader parseAs(capnp::Type type, YAML::Node src) {
		KJ_REQUIRE(!type.isData());
		KJ_REQUIRE(!type.isStruct());
		KJ_REQUIRE(!type.isInterface());
		KJ_REQUIRE(!type.isList());
		
		if(type.isEnum()) {
			return type.asEnum().getEnumerant(src.as<std::string>());
		}
		
		if(type.isVoid()) return capnp::Void();
		
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
		
		HANDLE_TYPE(isText(), const char*);
		#undef HANDLE_TYPE
		
		return capnp::Void();
	}
	
	void parseAs(YAML::Emitter emitter, DynamicValue::Reader val) {
		auto type = val.getType();
		
		KJ_REQUIRE(!type.isData());
		KJ_REQUIRE(!type.isStruct());
		KJ_REQUIRE(!type.isInterface());
		KJ_REQUIRE(!type.isList());
		
		if(type.isEnum()) {
			emitter << kj::str(val).cStr();
		}
		
		if(type.isVoid()) return capnp::Void();
		
		if(type.isText())) {
			emitter << val.as<capnp::Text>().cStr();
		}
		
		#define HANDLE_TYPE(checkFun, tp) \
			if(type.checkFun()) emitter << val.as<tp>();
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
	}
	
	void load(capnp::DynamicList::Builder dst, YAML::Node src) {
		auto listSchema = dst.getSchema();
		auto elType = listSchema.getElementType();
		
		for(auto i : kj::indices(dst.size)) {
			YAML::Node val = src[i];
			
			if(elType.isStruct()) {
				load(dst[i], val);
			} else if(elType.isList()) {
				auto out = dst.init(i, val.size());
				load(out.as<capnp::DynamicList>(), val);
			} else {
				dst.set(i, parseAs(elType, val));
			}
		}
	}
	
	void operator<<(YAML::Emitter& dst, capnp::DynamicList::Builder src) {
		out << YAML::BeginMap;
				
		out << YAML::EndMap();
	}
}

namespace fsc {

void load(capnp::DynamicStruct::Builder dst, YAML::Node src) {
	// Interpret nodes as map
	for(auto it = src.begin(); it != src.end(); ++it) {
		// Extract key and value
		std::string key = it -> first;
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
			
			KJ_REQUIRE(!type.isAnyPointer(), "Can not specify AnyPointer-typed fields with YAML");
			KJ_REQUIRE(!type.isInterface(), "Interface capabilities can not be set via YAML");
			
			if(type.isStruct()) {
				auto fieldVal = dst.init(field).as<capnp::DynamicStruct>();
				load(fieldVal, src);
				return;
			}
			if(type.isList()) {
				auto fieldVal = dst.init(field, val.size()).as<capnp::DynamicList>();
				load(fieldVal, src);
				return;
			}
			
			dst.set(field, parseAs(type, val));
		}
	}
}

}