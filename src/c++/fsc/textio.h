#pragma once

#include "data.h"

namespace fsc { namespace textio {
	struct Visitor {
		inline virtual ~Visitor() {};
		virtual capnp::Type expectedType() = 0;
		
		virtual void beginObject(Maybe<size_t>) = 0;
		virtual void endObject() = 0;
		
		virtual void beginArray(Maybe<size_t>) = 0;
		virtual void endArray() = 0;
		
		virtual void accept(DynamicValue::Reader value) = 0;
		virtual void acceptKey(kj::StringPtr key) = 0;
	};
	
	struct Dialect {
		enum Language {
			JSON, CBOR, BSON, YAML
		};
		
		Language language = JSON;
		
		bool quoteSpecialNums = true;
		kj::StringPtr jsonInf = "inf";
		kj::StringPtr jsonNegInf = "-.inf";
		kj::StringPtr jsonNan = ".nan";
		
		inline bool isBinary() {
			switch(language) {
				case CBOR:
				case BSON:
					return true;
				
				case JSON:
				case YAML:
					return false;
			}
		}
	};
	
	using ListInitializer = kj::Function<capnp::DynamicList::Builder(size_t)>;
	
	void load(kj::BufferedInputStream&, Visitor&, const Dialect&);
	
	Own<Visitor> createVisitor(DynamicStruct::Builder);
	Own<Visitor> createVisitor(capnp::ListSchema, ListInitializer);
	
	void load(kj::BufferedInputStream&, DynamicStruct::Builder, const Dialect&);
	void load(kj::BufferedInputStream&, capnp::ListSchema, ListInitializer, const Dialect&);
	
	void save(DynamicValue::Reader, Visitor&);
	void save(DynamicValue::Reader, kj::BufferedOutputStream&, const Dialect&);
	
	namespace internal {
		void jsonconsLoad(kj::BufferedInputStream&, Visitor&, const Dialect&);
		Own<Visitor> createJsonconsWriter(kj::BufferedOutputStream&, const Dialect&);
		
		void yamlcppLoad(kj::BufferedInputStream&, Visitor&, const Dialect&);
		Own<Visitor> createYamlcppWriter(kj::BufferedOutputStream&, const Dialect&);
	}
}}