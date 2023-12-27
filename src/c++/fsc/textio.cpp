#include "textio.h"

using capnp::DynamicList;
using capnp::DynamicValue;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;

namespace fsc { namespace textio {

namespace {
	
	struct Sink {
		virtual ~Sink() {};
		
		virtual DynamicStruct::Builder newObject() = 0;
		virtual ListInitializer newList() = 0;
		virtual void accept(Orphan<DynamicValue>) = 0;
		virtual capnp::Type expectedType() = 0;
		virtual void finish() = 0;
		
		virtual void acceptKey(kj::StringPtr) = 0;
		
		capnp::Orphanage orphanage() = 0;
	};
	
	struct ExternalStructSink : public Sink {
		DynamicStruct::Builder builder;
		bool consumed = false;
		
		DynamicStruct::Builder newObject() override {
			KJ_REQUIRE(!consumed, "Internal error");
			consumed = true;
			return builder;
		}
		
		Sink::ListInitializer newList() override {
			KJ_FAIL_REQUIRE("Internal error");
		}
		
		void finish() override { KJ_FAIL_REQUIRE("Can not finish external slot"); }
		void accept(DynamicValue::Reader) override {
			KJ_FAIL_REQUIRE("Internal error");
		}
		
		capnp::Type expectedType() override {
			return builder.getSchema();
		}
		
		void acceptKey(kj::StringPtr) override {
			KJ_FAIL_REQUIRE("Internal error");
		}
	};
			
	struct ExternalListSink : public Sink {
		capnp::ListSchema listSchema;
		ListInitializer initializer;
		bool consumed = false;
		
		ExternalListSink(ListSchema ls, ListInitializer&& initer) :
			listSchema(ls), initializer(mv(initer))
		{}
		
		DynamicStruct::Builder newObject() override {
			KJ_FAIL_REQUIRE("Can not assign a map to an array");
		}
		
		ListInitializer newList() override {
			KJ_REQUIRE(!consumed, "Internal error");
			consumed = true;
			return mv(initializer);
		}
		
		void finish() override { KJ_FAIL_REQUIRE("Internal error"); }
		void accept(DynamicValue::Reader) override {
			KJ_FAIL_REQUIRE("Internal error");
		}
		
		capnp::Type expectedType() {
			return listSchema;
		}
		
		void acceptKey(kj::StringPtr) override {
			KJ_FAIL_REQUIRE("Internal error");
		}
	};
	
	struct StructSink : public Sink {
		DynamicStruct::Builder builder;
		Maybe<capnp::StructSchema::Field> currentField = nullptr;
		Maybe<kj::String> previouslySetUnionField = nullptr;
		
		StructSink(DynamicStruct::Builder builder) :
			builder(builder)
		{}
		
		// WARNING: The builder returned is only valid until the next call to this
		// object
		DynamicStruct::Builder newObject() override {
			KJ_IF_MAYBE(pField, currentField) {
				auto field = *pField;
				currentField = nullptr;
				KJ_REQUIRE(field.getType().isStruct(), "Can only decode maps as structs");
				
				return dst.init(field);
			}
			KJ_FAIL_REQUIRE("Trying to allocate map entry without key");
		}
		
		// WARNING: The initializer returned is only valid until the next call to this
		// object
		ListInitializer newList() override {
			KJ_IF_MAYBE(pField, currentField) {
				auto field = *pField;
				currentField = nullptr;
				
				KJ_REQUIRE(field.getType().isList(), "Can only decode arrays as lists");
				
				return [builder, field](size_t size) {
					return builder.init(field, size);
				};
			}
			KJ_FAIL_REQUIRE("Trying to allocate map entry without key");
		}
		
		void accept(DynamicValue::Builder val) override {
			KJ_IF_MAYBE(pField, currentField) {
				auto field = *pField;
				currentField = nullptr;
				
				if(val.getType() == DynamicValue::VOID)
					builder.clear(field);
				else
					builder.set(field, val);
			}
			KJ_FAIL_REQUIRE("Trying to set map entry without key");
		}
		
		capnp::Type expectedType() override {
			KJ_IF_MAYBE(pField, currentField) {
				return pField -> getType();
			}
			
			// Key types are text
			return capnp::Type(capnp::schema::Type::TEXT);
		}
		
		void acceptKey(kj::StringPtr key) override {
			KJ_REQUIRE(currentField == nullptr, "Can only select one field at once");
			
			// Make sure we don't set union fields twice
			if(field.getProto().getDiscriminantValue() != capnp::schema::Field::NO_DISCRIMINANT) {
				KJ_IF_MAYBE(pPrevSet, previouslySetUnionField) {
					kj::StringPtr currentField = key;
					kj::StringPtr previousField = *pPrevSet;
					
					KJ_FAIL_REQUIRE("Can not set two fields of the same union", currentField, previousField);
				}
				previouslySetUnionField = kj::str(key);
			}
			
			currentField = builder.getFieldByName(key);
		}
		
		void beginList(Maybe<size_t>) override {
			KJ_FAIL_REQUIRE("Can not assign a list to a struct slot");
		}
		
		void beginObject(Maybe<size_t>) override {}
		void finish() override {}
		
		capnp::Orphanage orphanage() override {
			return capnp::Orphanage::getForMessageContaining(builder);
		}
	};
	
	struct ListSink {
		ListInitializer initializer;
		capnp::ListSchema listSchema;
		
		capnp::MallocMessageBuilder tmpMessage;
		
		using TmpList = capnp::Orphan<capnp::DynamicList>;
		kj::Vector<TmpList> finishedLists;
		TmpList currentStorage;
		
		capnp::DynamicList::Builder currentList;
		size_t currentListComplete = 0;
		bool preInitialized = false;
		
		ListSink(capnp::ListSchema listSchema, ListInitializer&& initializer, Maybe<size_t> maybeSize) :
				initializer(mv(initializer))
				listSchema(listSchema))
		{
			KJ_IF_MAYBE(pSize, maybeSize) {
				size_t size = *pSize;
				currentList = initializer(size);
				preInitialized = true;
			} else {
				auto elType = listSchema.getElementType();
				size_t size = elType.isBool() ? 64 : 8;
				currentStorage = tmpMessage.getOrphanage().newOrphan(listSchema, size);
				currentList = currentStorage.get();
			}
		}
		
		void finish() override {
			if(preInitialized) {
				KJ_REQUIRE(currentListComplete == currentList.size(), "The number of elements presented in the array was smaller than initially advertised");
				return;
			}
			
			// Initialize target orphan
			size_t totalSize = 0;
			for(auto& f : finishedLists)
				totalSize += f.get().size();
			totalSize += currentListComplete;
			
			// Copy all data over
			DynamicList::Builder builder = initializer(totalSize);
			
			size_t outIdx = 0;
			for(auto& f : finishedLists) {
				auto reader = f.getReader();
				for(auto el : reader)
					builder.set(outIdx++, el);
			}
			
			for(auto i : kj::range(0, currentListComplete)) {
				builder.set(outIdx++, currentList[i]);
			}
		}
		
		// WARNING: The builder returned is only valid until the next call to this
		// object
		DynamicStruct::Builder newObject() override {
			grow();
			return currentList.get()[currentListComplete++];
		}
		
		// WARNING: The initializer returned is only valid until the next call to this
		// object
		ListInitializer newList() override {
			grow();
			return [dst = currentList.get(), offset = currentListComplete++](size_t size) {
				return dst.init(offset, size);
			};
		}
		
		void accept(DynamicValue::Builder val) override {
			grow();
			auto dst = currentList.get();
			
			if(val.getType() == DynamicValue::VOID)
				dst.clear(currentListComplete++);
			else
				dst.set(currentListComplete++, val);
		}
		
		capnp::Type expectedType() override {
			return listSchema.getElementType();
		}
		
		void acceptKey(kj::StringPtr) override {
			KJ_FAIL_REQUIRE("Can not assign keys to lists, only in objects");
		}
		
		capnp::Orphanage orphanage() override {
			return capnp::Orphanage::getForMessageContaining(dst);
		}
		
	private:
		void grow() {
			size_t currentSize = currentList.size();
			if(currentListComplete >= currentSize) {
				KJ_REQUIRE(!preInitialized, "The number of elements presented exceeded the advertised number");
				
				finishedLists.append(mv(currentStorage));
				currentStorage = tmpMessage.getOrphanage().newOrphan(listSchema, 2 * currentSize);
				currentList = currentStorage.get();
				currentListComplete = 0;
			}
		}
	};
	
	struct BuilderStack : public Visitor {
		kj::Vector<Own<Sink>> stack;
		bool firstItem = false;
		
		BuilderStack(Own<Sink>&& first) {
			stack.append(mv(first));
		}
		
		capnp::Type expectedType() override {
			KJ_ASSERT(!stack.empty());
			
			return stack.back().expectedType();
		}
		
		void beginObject(Maybe<size_t>) override {
			KJ_ASSERT(!stack.empty());
			
			stack.append(kj::heap<StructSink>(stack.back().newObject()));
		}
		
		void endObject() override {
			finish();
		}
		
		void beginArray(Maybe<size_t> size) override {			
			KJ_ASSERT(!stack.empty());
			
			stack.append(kj::heap<ListSink>(elType.asList(), stack.back().newList(), size));
		}
		
		void endArray() override {
			finish();
		}
		
		void accept(capnp::DynamicValue::Reader input) override {
			KJ_ASSERT(!stack.empty());
			stack.back().accept(input);
		}
	};
	
	/* This class implements three important conversions:
	   - Contains ways of constructing most value types from texts
	   - Supports the shorthand notation "field" for {"field" : null} on structs
	   - Allows placeholder objects to be used for capabilities (that might later become meaningful)
	*/
	struct ValueConverter : public Visitor {
		Visitor& backend;
		size_t ignoreDepth = 0;
		
		ValueConverter(Visitor& nBack) :
			backend(nBack)
		{}
		
		capnp::Type expectedType() override {
			if(ignoreDepth > 0)
				return capnp::schema::Type::VOID;
			
			return backend.expectedType();
		}
		
		void beginObject(Maybe<size_t> s) override {
			if(ignoreDepth > 0 || ignoreBegin()) {
				++ignoreDepth;
				return;
			}
			
			backend.beginObject(s);
		}
		
		void endObject() override {
			if(ignoreDepth > 0) {
				--ignoreDepth;
				return;
			}
			
			backend.endObject();
		}
		
		void beginArray(Maybe<size_t> size) override {
			if(ignoreDepth > 0 || ignoreBegin()) {
				++ignoreDepth;
				return;
			}
			
			backend.beginArray(size);
		}
		
		void accept(DynamicValue::Reader input) override {
			auto type = expectedType();
			
			using ST = capnp::schema::Type;
			switch(type.which()) {
				case ST::DATA:
				case ST::TEXT:
				case ST::LIST:
				case ST::ANY_POINTER:
				case ST::VOID:
					backend.accept(input);
					break;
					
				case ST::ENUM: {
					uint16_t rawValue = 0;
					auto asEnum = type.asEnum();
					
					switch(input.getType()) {
						case DynamicValue::TEXT:
							rawValue = asEnum.getEnumerantByName(input.getAs<capnp::Text>()).getRaw();
							break;
						case DynamicValue::INT:
						case DynamicValue::UINT:
							rawValue = input.getAs<uint16_t>();
							break;
						default:
							KJ_FAIL_REQUIRE("Could not convert to enumerant", input);
					}
					
					backend.accept(DynamicEnum(asEnum, rawValue));
					break;
				}
				
				case ST::STRUCT: {
					if(input.getType() == DynamicValue::TEXT) {
						backend.beginObject();
						backend.acceptKey(input.as<capnp::Text>());
						backend.accept(capnp::Void());
						backend.endObject();
					} else {
						backend.accept(input);
					}
					break;
				};
				
				case ST::FLOAT32:
				case ST::FLOAT64:
					// Custom parsing for String values
					if(input.getType() == DynamicValue::TEXT) {
						kj::String asString = kj::heapString(input.as<capnp::Text>());
						for(char& c : asString)
							c = std::tolower(c);
						
						#define HANDLE_VAL(val, result) \
							if(asString == val) { \
								back.accept(result); \
								break; \
							}
							
						HANDLE_VAL(".nan", std::numeric_limits<double>::quiet_NaN())
						HANDLE_VAL(".inf", std::numeric_limits<double>::infinity())
						HANDLE_VAL("-.inf", -std::numeric_limits<double>::infinity())
						
						#define HANDLE_VAL(size, val, result) \
							if(asString.substr(0, size) == val) { \
								back.accept(result); \
								break; \
							}
							
						HANDLE_VAL(3, "nan", std::numeric_limits<double>::quiet_NaN())
						HANDLE_VAL(3, "inf", std::numeric_limits<double>::infinity())
						HANDLE_VAL(4, "-inf", -std::numeric_limits<double>::infinity())
						
						#undef HANDLE_VAL
									
						backend.accept(std::stod(asString));
					} else {
						backend.accept(input);
					}
					break;
				
				case ST::UINT8:
				case ST::UINT16:
				case ST::UINT32:
				case ST::UINT64: {
					if(input.getType() == DynamicValue::TEXT) {
						backend.accept(input.as<capnp::Text>().parseAs<uint64_t>());
					} else {
						backend.accept(input);
					}
					break;
				}
				
				case ST::INT8:
				case ST::INT16:
				case ST::INT32:
				case ST::INT64: {
					if(input.getType() == DynamicValue::TEXT) {
						back.accept(input.as<capnp::Text>().parseAs<int64_t>());
					} else {
						back.accept(input);
					}
					break;
				}
				
				case ST::BOOL:
					// Custom parsing for String values
					if(input.getType() == DynamicValue::TEXT) {
						kj::String asString = kj::heapString(input.as<capnp::Text>());
						for(char& c : asString)
							c = std::tolower(c);
						
						#define HANDLE_VAL(val, result) \
							if(asString == val) { \
								backend.accept(result); \
								break; \
							}
							
						HANDLE_VAL("true", true);
						HANDLE_VAL("false", false);
						HANDLE_VAL(".true", true);
						HANDLE_VAL(".false", false);
						HANDLE_VAL("yes", true);
						HANDLE_VAL("no", false);
						HANDLE_VAL("on", true);
						HANDLE_VAL("off", false);
						
						#undef HANDLE_VAL
						
						KJ_FAIL_REQUIRE("Unable to convert value to bool", asString);
					} else {
						backend.accept(input);
					}
					break;
				
				case ST::INTERFACE:
					if(input.getType() == DynamicValue::TEXT) {
						capnp::Capability::Client brokenCap(KJ_EXCEPTION(FAILED, input.getAs<capnp::Text>()));
						backend.accept(brokenCap.castAs<DynamicCapability>(type.asInterface()));
					}
					break;
			}	
		}
	
	private:
		void finish() {
			KJ_REQUIRE(stack.size() >= 2, "endObject or endArray called without matching beginObject or beginArray");
			stack.back().finish();
			stack.removeLast();
		}
		
		bool ignoreBegin() {
			auto tp = expectedType();
			
			if(tp.isVoid() || tp.isAnyPointer()) {
				backend.accept(capnp::Void());
				return true;
			} else if(tp.isCapability()) {
				capnp::Capability::Client brokenCap(KJ_EXCEPTION(FAILED, "Can not restore capability"));
				backend.accept(brokenCap.castAs<DynamicCapability>(tp.asInterface()));
				return true;
			}
			
			return false;
		}
	};
	
	void saveValue(DynamicValue::Reader, Visitor&);
	void saveStruct(DynamicStruct::Reader, Visitor&);
	void saveList(DynamicList::Reader, Visitor&);
	
	void saveValue(DynamicValue::Reader in, Visitor& v) {
		switch(in.getType()) {
			case DynamicValue::STRUCT:
				saveStruct(in.as<DynamicStruct>(), v);
				break;
			case DynamicValue::LIST:
				saveList(in.as<DynamicList>(), v);
				break;
			case DynamicValue::CAPABILITY: {
				auto asCap = in.as<DynamicCapability>();
				auto hook = capnp::ClientHook::from(kj::mv(asCap));
				if(hook -> isNull())
					v.accept(capnp::Void());
				else
					v.accept(in);
				break;
			}
			default:
				v.accept(in);
		}
	}
	
	void saveStruct(DynamicStruct::Reader in, Visitor& v) {
		// If we have no non-union fields and the active union field is a
		// default-valued one, then we write only the field name (reads nicer
		
		if(structSchema.getNonUnionFields().size() == 0) {
			auto maybeActive = in.which();
			KJ_IF_MAYBE(pActive, maybeActive) {
				auto& active = *pActive;
				if(!in.has(active, capnp::HasMode::NON_DEFAULT)) {
					capnp::Text::Reader fieldName = active.getProto().getName();
					v.accept(fieldName);
					return;
				}
			}
		}
		
		auto shouldEmitField = [&](capnp::StructSchema::Field field) {
			// Check if field is inactive union field
			if(field.getProto().getDiscriminantValue() != capnp::schema::Field::NO_DISCRIMINANT) {
				KJ_IF_MAYBE(pField, src.which()) {
					return *pField == field;
				}
				return false;
			} else {
				// No point in emitting void non-union fields
				if(field.getType().isVoid())
					return false;
			}
			
			// Don't emit null-valued fields for capabilities & AnyPointer
			if(field.getType().isAnyPointer() || field.getType().isCapability()) {
				if(!in.has(active, capnp::HasMode::NON_DEFAULT))
					return false;
			}
			
			return true;
		};
		
		size_t fieldCount = 0;
		for(auto field : structSchema.getFields()) {
			if(shouldEmitField(field))
				++fieldCount;
		}
		
		v.beginObject(fieldCount);
		
		auto emitField = [&](capnp::StructSchema::Field field) {
			auto type = field.getType();
			
			v.acceptKey(field.getProto().getName());
			saveValue(src.get(field), v);
		};
		
		for(auto field : structSchema.getFields()) {
			if(shouldEmitField(field))
				emitField(field);
		}
		
		v.endObject();
	}
	
	void saveList(DynamicList::Reader list, Visitor& v) {
		v.beginList(list.size());
		
		for(auto el : list)
			saveValue(el, v);
		
		v.endList();
	}
	
	Visitor makeBuilderStack(Own<Sink>&& sink) {
		auto stack = kj::heap<BuilderStack>(mv(sink));
		auto conv = kj::heap<ValueConverter>(*stack);
		return conv.attach(stack);
	}
}

Own<Visitor> makeVisitor(DynamicStruct::Builder b) {
	return makeBuilderStack(kj::heap<ExternalStructSink>(b));
}

Own<Visitor> makeVisitor(capnp::ListSchema schema, ListInitializer initializer) {
	return makeBuilderStack(kj::heap<ExternalListSink>(schema, mv(initializer)));
}

void load(kj::BufferedInputStream& is, DynamicStruct::Builder builder, const Dialect& dialect) {
	BuilderStack stack(kj::heap<ExternalStructSink>(builder));
	ValueConverter conv(stack);
	load(is, conv, dialect);
}

void load(kj::BufferedInputStream& is, capnp::ListSchema schema, ListInitializer initializer, const Dialect&) {
	BuilderStack stack(kj::heap<ExternalListSink>({schema, mv(initializer)});
	ValueConverter conv(stack);
	load(is, conv, dialect);
}

void load(kj::BufferedInputStream& is, Visitor& visitor, const Dialect& dialect) {
	if(dialect.language == Dialect::YAML) {
		yamlcppLoad(is, visitor, dialect);
	} else {
		jsonconsLoad(is, visitor, dialect);
	}
}

void save(DynamicValue::Reader reader, Visitor& v) {
	saveValue(reader, v);
}

void save(DynamicValue::Reader reader, kj::BufferedOutputStream& os, Dialect& dialect) {
	Own<Visitor> v;
	
	if(dialect.language == Dialect::YAML) {
		v = internal::createYamlcppWriter(os, dialect);
	} else {
		v = internal::createJsonconsWriter(os, dialect);
	}
	
	save(reader, *v);
}

}}