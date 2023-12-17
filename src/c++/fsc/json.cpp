#include "json.h"

#include <goldfish/json_reader.h>
#include <goldfish/json_writer.h>
#include <goldfish/cbor_reader.h>
#include <goldfish/cbor_writer.h>
#include <goldfish/stream.h>

using capnp::DynamicList;
using capnp::DynamicValue;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;

// This file is based on yaml.cpp, but adapted to use the goldfish CBOR / JSON writer.

namespace fsc {

namespace {
	struct GoldfishInputStream {
		kj::InputStream& backend;
		
		GoldfishInputStream(kj::InputStream& backend) :
			backend(backend)
		{}
		
		size_t read_partial_buffer(goldfish::buffer_ref ref) {
			void* data = ref.data();
			size_t size = ref.size();
			
			return backend.read(data, 0, size);
		}
	};
	
	struct GoldfishOutputStream {
		kj::OutputStream& backend;
		
		GoldfishOutputStream(kj::OutputStream& backend) :
			backend(backend)
		{}
		
		void write_buffer(goldfish::const_buffer_ref ref) {
			const void* data = ref.data();
			size_t size = ref.size();
			
			backend.write(data, size);
		}
		
		size_t flush() {
			return 0;
		}
	};
	
	template<typename Reader>
	DynamicValue::Reader parsePrimitive(capnp::Type type, Reader& reader) {
		KJ_REQUIRE(!type.isData());
		KJ_REQUIRE(!type.isText());
		KJ_REQUIRE(!type.isStruct());
		KJ_REQUIRE(!type.isList());
		
		if(type.isEnum()) {
			KJ_IF_MAYBE(pEnumerant, type.asEnum().findEnumerantByName(reader.as_string())) {
				return DynamicEnum(*pEnumerant);
			} else {
				return DynamicEnum(type.asEnum(), reader.as_uint16());
			}
		}
		
		if(type.isVoid()) return capnp::Void();		

		if(type.isAnyPointer()) return nullptr;
		if(type.isInterface()) return nullptr;
		
		if((type.isFloat32() || type.isFloat64()) && reader.is_exactly<goldfish::tags::string>()) {			
			// We also allow string to number conversion
			std::string asString = reader.as_string();
			for(unsigned char& c : asString)
				c = std::tolower(c);
			
			if(asString == ".nan")
				return std::numeric_limits<double>::quiet_NaN();
			if(asString == ".inf")
				return std::numeric_limits<double>::infinity();
			if(asString == "-.inf")
				return -std::numeric_limits<double>::infinity();
						
			return std::stod(asString);
		}
		
		#define HANDLE_TYPE(checkFun, convFun) \
			if(type.checkFun()) return reader.convFun();
		
		HANDLE_TYPE(isBool, as_bool);
		HANDLE_TYPE(isFloat32, as_double);
		HANDLE_TYPE(isFloat64, as_double);
		
		HANDLE_TYPE(isInt8, as_int8);
		HANDLE_TYPE(isInt16, as_int16);
		HANDLE_TYPE(isInt32, as_int32);
		HANDLE_TYPE(isInt64, as_int64);
		
		HANDLE_TYPE(isUInt8, as_uint8);
		HANDLE_TYPE(isUInt16, as_uint16);
		HANDLE_TYPE(isUInt32, as_uint32);
		HANDLE_TYPE(isUInt64, as_uint64);
		#undef HANDLE_TYPE
		
		return capnp::Void();
	}
	
	template<typename Writer>
	void emitPrimitive(DynamicValue::Reader val, Writer& writer, bool strict) {
		auto type = val.getType();
				
		KJ_REQUIRE(type != DynamicValue::STRUCT);
		KJ_REQUIRE(type != DynamicValue::LIST);
		
		switch(type) {
			case DynamicValue::CAPABILITY: {
				auto asCap = val.as<DynamicCapability>();
				auto hook = capnp::ClientHook::from(kj::mv(asCap));
				if(hook -> isNull())
					writer.write(nullptr);
				else
					writer.write("<capability>");
				
				break;
			}
			case DynamicValue::ANY_POINTER:
				writer.write("<unknown>");
				break;
				
			case DynamicValue::VOID:
				writer.write(nullptr);
				break;
				
			case DynamicValue::DATA: {
				auto asBin = val.as<capnp::Data>();
				
				writer.startBinary(asBin.size());
				writer.write(goldfish::const_buffer_ref(asBin.begin(), asBin.size()));
				break;
			}
			
			case DynamicValue::TEXT:
				writer.write(val.as<capnp::Text>().cStr());
				break;
				
			case DynamicValue::ENUM: {
				auto enumerant = val.as<DynamicEnum>().getEnumerant();
				KJ_IF_MAYBE(pEn, enumerant) {
					writer.write(pEn -> getProto().getName().cStr());
				} else {
					writer.write(val.as<DynamicEnum>().getRaw());
				}
				break;
			}
			
			case DynamicValue::BOOL:
				writer.write(val.as<bool>());
				break;
			
			case DynamicValue::FLOAT: {
				double dVal = val.as<double>();
				if(!std::isfinite(dVal)) {
					if(dVal > 0) {
						writer.write(".inf");
					} else if(dVal < 0) {
						writer.write("-.inf");
					} else {
						writer.write(".nan");
					}
				} else {
					writer.write(dVal);
				}
				break;
			}
			
			case DynamicValue::INT:
				writer.write(val.as<int64_t>());
				break;
			
			case DynamicValue::UINT:
				writer.write(val.as<uint64_t>());
				break;
			
			default:
				KJ_FAIL_REQUIRE("Internal error: Unhandled type");
		}
	}
	
	template<typename ListInitializer, typename Reader>
	void loadList(ListInitializer initializer, Reader& reader) {
		// Unfortunately, we do not know the size of the list beforehand in JSON, but we need
		// to know the list size to allocate the output buffer. There is only one solution for
		// this: Allocate a temporary document, dump the input there, then scan that document.
		
		// Step 1: Dump to temporary document & determine list size
		size_t listSize = 0;
		auto tmpWriter = goldfish::cbor::create_writer(
			goldfish::stream::vector_writer()
		).begin_array();
		
		{
			auto inArray = reader.as_array();
			while(true) {
				auto optVal = inArray.read();
				if(!optVal)
					break;
				
				tmpWriter.write(*optVal);
				++listSize;
			}
		}
		
		std::vector<char> tmpDocument = tmpWriter.flush();
		
		// Step 2: Allocate list
		DynamicList::Builder dst = initializer(listSize);
		
		// Step 3: Load list
		auto listReader = goldfish::cbor::create_reader(
			goldfish::stream::read_vector(mv(tmpDocument))
		).as_array();
		
		auto listSchema = dst.getSchema();
		auto elType = listSchema.getElementType();
		
		for(auto i : kj::indices(dst)) {
			auto valReader = listReader.read();
			
			if(elType.isStruct()) {
				loadStruct(dst[i].as<DynamicStruct>(), *valReader);
			} else if(elType.isList()) {
				auto initializer = [&](size_t listSize) {
					return dst.init(i, listSize);
				};
				loadList(initializer, *valReader);
			} else if(elType.isData()) {
				auto asBinary = valReader -> as_binary();
				capnp::Data::Reader dataPtr(asBinary.data(), asBinary.size());
				dst.set(i, dataPtr);
			} else if(elType.isText()) {
				auto asString = valReader -> as_string();
				dst.set(i, capnp::Text::Reader(asString));
			} else {
				dst.set(i, parsePrimitive(elType, *valReader));
			}
		}
	}
	
	template<typename Reader>
	void loadStruct(DynamicStruct::Builder dst, Reader& reader) {
		// If the node is scalar, this means we set that field with default value
		if(!reader.is_exactly<goldfish::map>()) {
			auto field = dst.getSchema().getFieldByName(reader.as_string());
			dst.clear(field);
			
			return;
		}
		
		Maybe<kj::String> previouslySetUnionField = nullptr;
		
		auto as_map = reader.as_map();
		
		while(true) {
			auto optionalKey = reader.read_key();
			if(!optionalKey)
				break;
			
			std::string key = optionalKey -> as_string();
			
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
				
				KJ_REQUIRE(!elType.isAnyPointer(), "Can not specify AnyPointer-typed fields with JSON/CBOR");
				KJ_REQUIRE(!elType.isInterface(), "Interface capabilities can not be set via JSON/CBOR");
				
				auto valueReader = reader.read_value();
				
				if(type.isStruct()) {
					auto fieldVal = dst.init(field).as<DynamicStruct>();
					loadStruct(fieldVal, valueReader);
					continue;
				}
				if(type.isList()) {
					auto initializer = [&](size_t listSize) {
						return dst.init(field, listSize).as<DynamicList>();
					};
					loadList(initializer, valueReader);
					
					continue;
				}
				if(type.isData()) {
					auto asBinary = reader.as_binary();
					capnp::Data::Reader dataPtr(asBinary.data(), asBinary.size());
					dst.set(field, dataPtr);
					continue;
				}
				if(type.isText()) {				
					auto asString = reader.as_string();
					dst.set(field, capnp::Text::Reader(asString));
					continue;
				}
				
				dst.set(field, parsePrimitive(type, reader));
			} else {
				KJ_LOG(WARNING, "Could not find matching field for map entry", key);
				// KJ_DBG(dst.getSchema(), kj::StringPtr(key));
			}
		}
	}
	
	template<typename Writer>
	void writeValue(DynamicValue::Reader src, Writer& writer, bool strictJson) {
		auto type = src.getType();
		
		if(type == DynamicValue::STRUCT) {
			writeStruct(src.as<DynamicStruct>(), writer, strictJson);
		} else if(type == DynamicValue::LIST) {
			writeList(src.as<DynamicList>(), writer, strictJson);
		} else {
			emitPrimitive(src, writer, strictJson);
		}
		
		return dst;
	}
	
	template<typename Writer>
	void writeList(DynamicList::Builder src, Writer& writer, bool strictJson) {
		auto asArray = writer.start_array();
		for(DynamicValue::Reader el : src)
			writeValue(el, asArray, strictJson);
	}
		
	template<typename Writer>
	void writeStruct(DynamicStruct::Builder src, Writer& writer, bool strictJson) {	
		auto structSchema = src.getSchema();
		
		// If we have no non-union fields and the active union field is a
		// default-valued one, then we write only the field name (reads nicer)
		if(structSchema.getNonUnionFields().size() == 0) {
			auto maybeActive = src.which();
			KJ_IF_MAYBE(pActive, maybeActive) {
				auto& active = *pActive;
				if(!src.has(active, capnp::HasMode::NON_DEFAULT)) {
					writer.write(active.getProto().getName().cStr());
					return;
				}
			}
		}
		
		auto asMap = writer.start_map();
		
		auto emitField = [&](capnp::StructSchema::Field field) {
			auto type = field.getType();
			
			// Inspect inner-most type for lists
			auto elType = type;
			while(elType.isList()) {
				auto asList = elType.asList();
				elType = asList.getElementType();
			}
			
			auto val = src.get(field);
			auto valWriter = asMap.append(field.getProto().getName().cStr());
			
			if(type.isStruct()) {
				writeStruct(val.as<DynamicStruct>(), valWriter, strictJson);
			} else if(type.isList()) {
				writeList(val.as<DynamicList>(), valWriter, strictJson);
			} else {
				emitPrimitive(val, valWriter, strictJson);
			}
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
	}
}

}