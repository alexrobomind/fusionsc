#pragma once

#include "data.h"

/*

This module translates between hierarchical formats with structured object
representations. Formats should support the following types:
- String-keyed maps
- Lists
- Null
- double, int64, uint64
- Strings
- Binary string

*/

namespace fsc { namespace textio {
	// Tree node that can hold our text IO structures
	struct Node {		
		// using MapPayload = kj::TreeMap<kj::String, Node>;
		using MapPayload = kj::Vector<kj::Tuple<Node, Node>>;
		using ListPayload = kj::Vector<Node>;
		struct NullValue {};
		
		using Payload = OneOf<
			MapPayload, ListPayload,
			kj::String, kj::Array<kj::byte>,
			double, uint64_t, int64_t, bool,
			NullValue
		>;
		
		Payload payload;
	};
	
	struct Visitor {
		inline virtual ~Visitor() noexcept(false) {};
		
		virtual void beginObject(Maybe<size_t>) = 0;
		virtual void endObject() = 0;
		
		virtual void beginArray(Maybe<size_t>) = 0;
		virtual void endArray() = 0;
		
		virtual void acceptNull() = 0;
		virtual void acceptDouble(double) = 0;
		virtual void acceptInt(int64_t) = 0;
		virtual void acceptUInt(uint64_t) = 0;
		virtual void acceptString(kj::StringPtr) = 0;
		virtual void acceptData(ArrayPtr<const byte>) = 0;
		virtual void acceptBool(bool) = 0;
		
		// virtual void acceptKey(kj::StringPtr) = 0;
		
		virtual bool done() = 0;
		
		//! Whether this visitor allows integer map keys
		bool supportsIntegerKeys = false;
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
		
		bool humanReadable = true;
		
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
	
	struct SaveOptions {
		/** Enable compact representation
		 *
		 * Compact representation allows hiding default valued fields
		 * and substituting structures of the form { "onlyField" : defaultValue }
		 * with the field key ("onlyField" or the field ID).
		 */
		bool compact = true;
		
		/** Enable integer field keys
		 *
		 * Some formats (YAML / CBOR) permit arbitrary objects to be used as
		 * keys. In such a case, we can save space and improve protcol
		 * migration by preferentially using the numeric IDs instead of names
		 * for a lot of objects (field keys, enumerants).
		 *
		 * Since this eliminates the value of the saved data as self-describing
		 * structures, it is disabled by default.
		 */
		 bool integerKeys = false;
	};
	
	using ListInitializer = kj::Function<capnp::DynamicList::Builder(size_t)>;
	
	Own<Visitor> createVisitor(capnp::DynamicStruct::Builder);
	Own<Visitor> createVisitor(capnp::ListSchema, ListInitializer);
	Own<Visitor> createVisitor(Node&);
	
	void load(kj::BufferedInputStream&, Visitor&, const Dialect&);
	void load(kj::BufferedInputStream&, Node&, const Dialect&);
	void load(kj::BufferedInputStream&, capnp::DynamicStruct::Builder, const Dialect&);
	void load(kj::BufferedInputStream&, capnp::ListSchema, ListInitializer, const Dialect&);
	
	void save(Node&, Visitor&);
	void save(capnp::DynamicValue::Reader, Visitor&, const SaveOptions& = SaveOptions());
	void save(capnp::DynamicValue::Reader, kj::BufferedOutputStream&, const Dialect&, const SaveOptions& = SaveOptions());
	
	namespace internal {
		void jsonconsLoad(kj::BufferedInputStream&, Visitor&, const Dialect&);
		Own<Visitor> createJsonconsWriter(kj::BufferedOutputStream&, const Dialect&);
		
		void yamlcppLoad(kj::BufferedInputStream&, Visitor&, const Dialect&);
		Own<Visitor> createYamlcppWriter(kj::BufferedOutputStream&, const Dialect&);
	}
}}