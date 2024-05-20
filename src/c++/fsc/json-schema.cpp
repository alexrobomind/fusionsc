#include "json-schema.h"

#include <kj/encoding.h>

namespace fsc {

using Visitor = structio::Visitor;

namespace {

void writeSchema(capnp::Schema s, structio::Visitor& v, kj::HashSet<capnp::Schema>& alreadyWritten);
void writeType(capnp::Type t, structio::Visitor& v, kj::HashSet<capnp::Schema>& alreadyWritten, bool root = false);

void writeEnumSchema(capnp::EnumSchema s, structio::Visitor& v, kj::HashSet<capnp::Schema>& alreadyWritten) {
	v.acceptString("anyOf");
	v.beginArray(nullptr);
		writeType(capnp::schema::Type::UINT16, v, alreadyWritten);
		
		v.beginObject(nullptr);
			v.acceptString("enum");
			v.beginArray(nullptr);
				for(auto e : s.getEnumerants()) {
					v.acceptString(e.getProto().getName());
				}
			v.endArray();
		v.endObject();
	v.endArray();
}

void writeStructSchema(capnp::StructSchema s, structio::Visitor& v, kj::HashSet<capnp::Schema>& alreadyWritten) {
	v.acceptString("anyOf");
	v.beginArray(nullptr);
		v.beginObject(nullptr);
			v.acceptString("enum");
			v.beginArray(nullptr);
			
			v.acceptNull();
			
			if(s.getUnionFields().size() != 0) {
				for(auto field : s.getUnionFields())
					v.acceptString(field.getProto().getName());
			} else if(s.getUnionFields().size() == 0 && s.getFields().size() == 1) {
				v.acceptString(s.getFields()[0].getProto().getName());
			}
			
			v.endArray();
		v.endObject();
		
		v.beginObject(nullptr);
			v.acceptString("type");
			v.acceptString("object");
			
			v.acceptString("properties");
			v.beginObject(nullptr);
				for(auto field : s.getFields()) {
					v.acceptString(field.getProto().getName());
					writeType(field.getType(), v, alreadyWritten);
				}
			v.endObject();
		v.endObject();
	v.endArray();
}

void writeInterfaceSchema(capnp::InterfaceSchema s, structio::Visitor& v, kj::HashSet<capnp::Schema>& alreadyWritten) {
	// Check if we are a DataRef schema or not
	constexpr uint64_t DR_ID = capnp::typeId<DataRef<capnp::AnyPointer>>();
	
	if(s.getProto().getId() != DR_ID) {
		// Not a DataRef. Only accept string
		v.acceptString("type");
		v.acceptString("string");
	} else {
		v.acceptString("anyOf");
		v.beginArray(nullptr);
			v.beginObject(nullptr);
				v.acceptString("type");
				v.acceptString("string");
			v.endObject();
			
			v.beginObject(nullptr);
				v.acceptString("type");
				v.acceptString("object");
				
				v.acceptString("properties");
				v.beginObject(nullptr);
					v.acceptString("data");
					writeType(s.getBrandArgumentsAtScope(DR_ID)[0], v, alreadyWritten);
				v.endObject();
			v.endObject();
		v.endArray();
	}
}

void writeListSchema(capnp::ListSchema s, structio::Visitor& v, kj::HashSet<capnp::Schema>& alreadyWritten) {	
	v.acceptString("type");
	v.acceptString("array");
	
	v.acceptString("items");
	writeType(s.getElementType(), v, alreadyWritten);
}

kj::String idFor(capnp::Schema s) {
	if(!s.isBranded()) {
		return kj::str(s.getUnqualifiedName(), "_", s.getProto().getId());
	}
	Temporary<capnp::schema::Brand> brand;
	extractBrand(s, brand);
	
	return kj::str(s.getUnqualifiedName(), "_", s.getProto().getId(), "_", kj::encodeBase64(capnp::canonicalize(brand.asReader()).asBytes()));
}

void writeSchema(capnp::Schema s, structio::Visitor& v, kj::HashSet<capnp::Schema>& alreadyWritten) {
	bool isGroup = s.getProto().isStruct() && s.getProto().getStruct().getIsGroup();
	if(!isGroup) {
		if(alreadyWritten.contains(s)) {
			v.acceptString("$ref");
			v.acceptString(kj::str("#", idFor(s)));
			
			return;
		}
		
		alreadyWritten.insert(s);
		
		v.acceptString("id");
		v.acceptString(idFor(s));
		
		v.acceptString("title");
		v.acceptString(s.getProto().getDisplayName());
	}
	
	if(s.getProto().isEnum()) {
		writeEnumSchema(s.asEnum(), v, alreadyWritten);
	} else if(s.getProto().isStruct()) {
		writeStructSchema(s.asStruct(), v, alreadyWritten);
	} else {
		KJ_FAIL_REQUIRE("Internal error: writeSchema called for invalid schema kind");
	}
}

template<typename T>
void writeUInt(kj::StringPtr name, Visitor& v) {
	v.acceptString("type");
	v.acceptString("integer");
	
	v.acceptString("title");
	v.acceptString(name);
	
	v.acceptString("minimum");
	v.acceptUInt(0);
	
	v.acceptString("maximum");
	v.acceptUInt((T) kj::maxValue);
}

template<typename T>
void writeInt(kj::StringPtr name, Visitor& v) {
	v.acceptString("type");
	v.acceptString("integer");
	
	v.acceptString("title");
	v.acceptString(name);
	
	v.acceptString("minimum");
	v.acceptInt((T) kj::minValue);
	
	v.acceptString("maximum");
	v.acceptInt((T) kj::maxValue);
}

void writeType(capnp::Type t, structio::Visitor& v, kj::HashSet<capnp::Schema>& alreadyWritten, bool root) {
	using T = capnp::schema::Type;
	
	v.beginObject(nullptr);
	
	if(root) {
		v.acceptString("$schema");
		v.acceptString("http://json-schema.org/draft-07/schema#");
	}
	
	switch(t.which()) {
		case T::VOID: {
			v.acceptString("type");
			v.acceptString("null");
			break;
		}
		case T::BOOL: {
			v.acceptString("type");
			v.acceptString("boolean");
			break;
		}
		
		#define HANDLE_TYPE(type, ctype) \
			case T::type: \
				writeInt<ctype>(#type, v); \
				break;
		
		HANDLE_TYPE(INT8,  int8_t);
		HANDLE_TYPE(INT16, int16_t);
		HANDLE_TYPE(INT32, int32_t);
		HANDLE_TYPE(INT64, int64_t);
		
		#undef HANDLE_TYPE
		
		#define HANDLE_TYPE(type, ctype) \
			case T::type: \
				writeUInt<ctype>(#type, v); \
				break;
		
		HANDLE_TYPE(UINT8,  uint8_t);
		HANDLE_TYPE(UINT16, uint16_t);
		HANDLE_TYPE(UINT32, uint32_t);
		HANDLE_TYPE(UINT64, uint64_t);
		
		#undef HANDLE_TYPE
		
		case T::FLOAT32:
		case T::FLOAT64: {
			v.acceptString("type");
			v.acceptString("number");
			break;
		}
		
		case T::TEXT: {
			v.acceptString("type");
			v.acceptString("string");
			break;
		}
		
		case T::DATA:
		case T::ANY_POINTER: {
			v.acceptString("type");
			v.acceptString("string");
			
			v.acceptString("contentEnconding");
			v.acceptString("base64");
			break;
		}
		
		case T::LIST: {
			writeListSchema(t.asList(), v, alreadyWritten);
			break;
		}
		
		case T::ENUM: {
			writeSchema(t.asEnum(), v, alreadyWritten);
			break;
		}
		
		case T::STRUCT: {
			writeSchema(t.asStruct(), v, alreadyWritten);
			break;
		}
		
		case T::INTERFACE: {
			writeInterfaceSchema(t.asInterface(), v, alreadyWritten);
			break;
		}
	}
	
	v.endObject();
}

}

void writeJsonSchema(capnp::Type t, structio::Visitor& v) {
	kj::HashSet<capnp::Schema> alreadyWritten;
	writeType(t, v, alreadyWritten, true);
}

}