#include "yaml.h"

using capnp::DynamicStruct;
using capnp::DynamicList;
using capnp::DynamicValue;

namespace fsc {
	
namespace {
	DynamicValue::Reader parsePrimitive(capnp::Type type, YAML::Node src) {
		KJ_REQUIRE(!type.isData());
		KJ_REQUIRE(!type.isText());
		KJ_REQUIRE(!type.isStruct());
		KJ_REQUIRE(!type.isList());
		
		if(type.isEnum()) {
			KJ_IF_MAYBE(pEnumerant, type.asEnum().findEnumerantByName(src.as<std::string>())) {
				return capnp::DynamicEnum(*pEnumerant);
			} else {
				return capnp::DynamicEnum(type.asEnum(), src.as<uint16_t>());
			}
		}
		
		if(type.isVoid()) return capnp::Void();		

		if(type.isAnyPointer()) return nullptr;
		if(type.isInterface()) return nullptr;
		
		if(type.isFloat32() || type.isFloat64()) {
			// Pre-check the number type for some non-YAML literals
			std::string asString = src.Scalar();
			
			for(char& c : asString)
				c = std::tolower(c);
			
			// yaml-cpp handles the builtin literals just fine
			/*if(asString == ".nan")
				return std::numeric_limits<double>::quiet_NaN();
			if(asString == ".inf")
				return std::numeric_limits<double>::infinity();
			if(asString == "-.inf")
				return -std::numeric_limits<double>::infinity();*/
			
			if(asString.substr(0, 3) == "nan")
				return std::numeric_limits<double>::quiet_NaN();
			if(asString.substr(0, 3) == "inf")
				return std::numeric_limits<double>::infinity();
			if(asString.substr(0, 4) == "-inf")
				return -std::numeric_limits<double>::infinity();
			
			return src.as<double>();
		}
		
		#define HANDLE_TYPE(checkFun, tp) \
			if(type.checkFun()) return src.as<tp>();
		HANDLE_TYPE(isBool, bool);
		
		HANDLE_TYPE(isInt8, int);
		HANDLE_TYPE(isInt16, int16_t);
		HANDLE_TYPE(isInt32, int32_t);
		HANDLE_TYPE(isInt64, int64_t);
		
		HANDLE_TYPE(isUInt8, unsigned int);
		HANDLE_TYPE(isUInt16, uint16_t);
		HANDLE_TYPE(isUInt32, uint32_t);
		HANDLE_TYPE(isUInt64, uint64_t);
		#undef HANDLE_TYPE
		
		return capnp::Void();
	}
	
	void emitPrimitive(YAML::Emitter& emitter, DynamicValue::Reader val) {
		auto type = val.getType();
				
		KJ_REQUIRE(type != DynamicValue::STRUCT);
		KJ_REQUIRE(type != DynamicValue::LIST);
		
		switch(type) {
			// TODO: DataRefs
			case DynamicValue::CAPABILITY: {
				auto asCap = val.as<capnp::DynamicCapability>();
				auto hook = capnp::ClientHook::from(kj::mv(asCap));
				if(hook -> isNull())
					emitter << "null";
				else
					emitter << "<capability>";
				
				break;
			}
			case DynamicValue::ANY_POINTER:
				emitter << "<unknown>";
				break;
				
			case DynamicValue::VOID:
				emitter << YAML::Null;
				break;
				
			case DynamicValue::DATA: {
				auto asBin = val.as<capnp::Data>();
				YAML::Binary data(asBin.begin(), asBin.size());
				emitter << data;
				break;
			}
			
			case DynamicValue::TEXT:
				emitter << val.as<capnp::Text>().cStr();
				break;
				
			case DynamicValue::ENUM: {
				auto enumerant = val.as<capnp::DynamicEnum>().getEnumerant();
				KJ_IF_MAYBE(pEn, enumerant) {
					emitter << pEn -> getProto().getName().cStr();
				} else {
					emitter << val.as<capnp::DynamicEnum>().getRaw();
				}
				break;
			}
			
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
	
	void emitList(YAML::Emitter& dst, DynamicList::Reader src) {
		auto listSchema = src.getSchema();
		auto elType = listSchema.getElementType();
		
		if(!elType.isStruct() && !elType.isList() && !elType.isData() && !elType.isText())
			dst << YAML::Flow;
		
		dst << YAML::BeginSeq;
		
		for(auto el : src) {
			dst << el;
		}
		
		dst << YAML::EndSeq;
	}
		
	void emitStruct(YAML::Emitter& dst, DynamicStruct::Reader src) {
		auto structSchema = src.getSchema();
		
		// If we have no non-union fields and the active union field is a
		// default-valued one, then we write only the field name (reads nicer)
		if(structSchema.getNonUnionFields().size() == 0) {
			auto maybeActive = src.which();
			KJ_IF_MAYBE(pActive, maybeActive) {
				auto& active = *pActive;
				if(!src.has(active, capnp::HasMode::NON_DEFAULT)) {
					dst << active.getProto().getName().cStr();
					return;
				}
			}
		}
		dst << YAML::BeginMap;
		
		auto emitField = [&](capnp::StructSchema::Field field) {
			auto type = field.getType();
			
			// Inspect inner-most type for lists
			auto elType = type;
			while(elType.isList()) {
				auto asList = elType.asList();
				elType = asList.getElementType();
			}
			
			auto val = src.get(field);
			
			dst << YAML::Key << field.getProto().getName().cStr();
			dst << YAML::Value << val;
		};
			
		for(auto field : structSchema.getNonUnionFields()) {
			// There is no point in emitting a non-union void field.
			if(field.getType().isVoid())
				continue;
			
			// Don't emit interface- or any-typed fields or nested lists
			if(field.getType().isInterface() || field.getType().isAnyPointer())
				continue;
			
			emitField(field);
		}
		
		KJ_IF_MAYBE(pField, src.which()) {
			emitField(*pField);
		}	
				
		dst << YAML::EndMap;
	}
}

capnp::DynamicValue::Reader loadPrimitive(capnp::Type type, YAML::Node src) {
	return parsePrimitive(mv(type), mv(src));
}
	
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
	// If the node is scalar, this means we set that field with default value
	if(src.IsScalar()) {
		auto field = dst.getSchema().getFieldByName(src.as<std::string>());
		dst.clear(field);
		
		return;
	}
		
	Maybe<kj::String> previouslySetUnionField = nullptr;
	
	// Interpret nodes as map
	for(auto it = src.begin(); it != src.end(); ++it) {
		// Extract key and value
		std::string key = it -> first.as<std::string>();
		YAML::Node val = it -> second;
		
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
				continue;
			}
			if(type.isList()) {
				auto fieldVal = dst.init(field, val.size()).as<capnp::DynamicList>();
				load(fieldVal, val);
				continue;
			}
			if(type.isData()) {
				auto asBinary = val.as<YAML::Binary>();
				capnp::Data::Reader dataPtr(asBinary.data(), asBinary.size());
				dst.set(field, dataPtr);
				continue;
			}
			if(type.isText()) {				
				auto asString = val.as<std::string>();
				dst.set(field, capnp::Text::Reader(asString));
				continue;
			}
			
			dst.set(field, parsePrimitive(type, val));
		} else {
			KJ_LOG(WARNING, "Could not find matching field for map entry", key);
			// KJ_DBG(dst.getSchema(), kj::StringPtr(key));
		}
	}
}

YAML::Emitter& operator<<(YAML::Emitter& dst, DynamicValue::Reader src) {
	auto type = src.getType();
	
	if(type == DynamicValue::STRUCT) {
		emitStruct(dst, src.as<DynamicStruct>());
	} else if(type == DynamicValue::LIST) {
		emitList(dst, src.as<DynamicList>());
	} else {
		emitPrimitive(dst, src);
	}
	
	return dst;
}

}