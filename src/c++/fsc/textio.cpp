#include "textio.h"

using capnp::DynamicList;
using capnp::DynamicValue;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;

namespace fsc { namespace textio {

namespace {
	void checkList(Maybe<capnp::Type> tp) {
		KJ_IF_MAYBE(pType, tp) {
			if(pType -> isList())
				return;
			
			if(pType -> isAnyPointer()) {
				auto apKind = pType -> whichAnyPointerKind();
				KJ_REQUIRE(apKind == capnp::schema::Type::AnyPointer::UNCONSTRAINED || apKind == capnp::schema::Type::AnyPointer::ANY_LIST);
				return ;
			}
			
			KJ_FAIL_REQUIRE("Type is not a valid list type");
		}
	}
	
	void checkStruct(Maybe<capnp::Type> tp) {
		KJ_IF_MAYBE(pType, tp) {
			if(pType -> isList())
				return;
			
			if(pType -> isAnyPointer()) {
				auto apKind = pType -> whichAnyPointerKind();
				KJ_REQUIRE(apKind == capnp::schema::Type::AnyPointer::UNCONSTRAINED || apKind == capnp::schema::Type::AnyPointer::ANY_LIST);
				return;
			}
			
			KJ_FAIL_REQUIRE("Type is not a valid list type");
		}
	}
	
	Maybe<capnp::Type> elementType(Maybe<capnp::Type> tp) {
		KJ_IF_MAYBE(pType, tp) {
			if(pType -> isList())
				return pType -> asList().getElementType();
		}
		
		return nullptr;
	}
	
	Maybe<capnp::Type> fieldType(Maybe<capnp::Type> tp, kj::StringPtr name) {
		KJ_IF_MAYBE(pType, tp) {
			if(pType -> isStruct()) {
				return pType -> asStruct().getFieldByName(name).getType();
			}
		}
		
		return nullptr;
	}
	
	struct Sink {
		virtual ~Sink() {};
		
		virtual DynamicStruct::Builder newObject() = 0;
		virtual DynamicList::Builder newList(size_t) = 0;
		virtual void accept(Orphan<DynamicValue>) = 0;
		virtual capnp::Type expectedType() = 0;
		
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
		
		ExternalListSink(capnp::ListSchema ls, ListInitializer&& initer) :
			listSchema(ls), initializer(mv(initer))
		{}
		
		DynamicStruct::Builder newObject() override {
			KJ_FAIL_REQUIRE("Can not assign a map to an array");
		}
		
		ListInitializer newList(size_t s) override {
			KJ_REQUIRE(!consumed, "Internal error");
			consumed = true;
			return initializer(s);
		}
		
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
		ListInitializer newList(size_t s) override {
			KJ_IF_MAYBE(pField, currentField) {
				auto field = *pField;
				currentField = nullptr;
				
				KJ_REQUIRE(field.getType().isList(), "Can only decode arrays as lists");
				return builder.init(field, size);
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
	};
	
	struct ListSink {
		DynamicList::Builder dst;
		size_t offset = 0;
		
		ListSink(DynamicList::Builder ndst) :
				dst(ndst)
		{}
		
		DynamicStruct::Builder newObject() override {
			return currentList.get()[offset++].as<DynamicStruct>();
		}
		
		ListInitializer newList(size_t s) override {
			return currentList.init(offset++, s).as<DynamicList>();
		}
		
		void accept(DynamicValue::Builder val) override {			
			if(val.getType() == DynamicValue::VOID)
				dst.clear(offset++);
			else
				dst.set(offset++, val);
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
	};
	
	struct BuilderStack {
		kj::Vector<Own<Sink>> stack;
		bool isDone = false;
		
		BuilderStack(Own<Sink>&& first) {
			stack.append(mv(first));
		}
		
		capnp::Type expectedType() {
			KJ_ASSERT(!stack.empty());
			
			return stack.back().expectedType();
		}
		
		void beginObject() {
			KJ_ASSERT(!isDone);
			
			stack.append(kj::heap<StructSink>(stack.back().newObject()));
		}
		
		void beginArray(size_t size) {			
			KJ_ASSERT(!isDone);
			
			stack.append(kj::heap<ListSink>(elType.asList(), stack.back().newList(), size));
		}
		
		void accept(capnp::DynamicValue::Reader input) {
			KJ_ASSERT(!isDone);
			stack.back().accept(input);
			
			if(stack.size() == 1)
				isDone = true;
		}
		
		void finish() {	
			KJ_ASSERT(!isDone);
			
			stack.back().finish();
			stack.removeLast();
			
			if(stack.size() == 1)
				isDone = true;
		}
	};
	
	/* This class implements three important conversions:
	   - Contains ways of constructing most value types from texts
	   - Supports the shorthand notation "field" for {"field" : null} on structs
	   - Allows placeholder objects to be used for capabilities (that might later become meaningful)
	*/
	struct ValueConverter : public Visitor {
		BuilderStack backend;
		size_t ignoreDepth = 0;
		
		struct Tape {
			Node dst;
			Own<Visitor> recorder;
			
			Tape() : recorder(createVisitor(dst)) {}
		};
		Maybe<Type> tape;
		
		ValueConverter(Own<Sink>&& firstSink) :
			backend(mv(firstSink))
		{}
		
		#define ACCEPT_FWD(expr) \
			if(ignoreDepth > 0) \
				return; \
			\
			KJ_IF_MAYBE(pTape, tape) { \
				pTape -> recorder -> expr; \
				\
				if(pTape -> recorder -> done() == 0) { \
					save(pTape -> node, *this); \
					tape = nullptr; \
				} \
				return; \
			}
		
		void beginObject(Maybe<size_t> s) override {
			ACCEPT_FWD(beginObject(s))
			
			if(ignoreDepth > 0 || ignoreBegin()) {
				++ignoreDepth;
				return;
			}
			
			backend.beginObject(s);
		}
		
		void endObject() override {
			ACCEPT_FWD(endObject())
			
			if(ignoreDepth > 0) {
				--ignoreDepth;
				return;
			}
			
			backend.finish();
		}
		
		void beginArray(Maybe<size_t> size) override {
			ACCEPT_FWD(beginArray(size))
			
			if(ignoreDepth > 0 || ignoreBegin()) {
				++ignoreDepth;
				return;
			}
			
			KJ_IF_MAYBE(pSize, size) {
				backend.beginArray(*pSize);
			} else {
				KJ_ASSERT(tape == nullptr, "Tape state error");
				
				Tape& newTape = tape.emplace();
				newTape.recorder.beginArray(nullptr);
				++newTape.depth;
			}
		}
		
		void endArray() {
			ACCEPT_FWD(endArray())
			
			if(ignoreDepth > 0) {
				--ignoreDepth;
				return;
			}
			
			backend.finish();
		}
		
		template<typename T>
		void acceptInteger(T i) {
			auto type = backend.expectedType();
			
			if(type.isEnum()) {
				KJ_REQUIRE(i >= 0);
				KJ_REQUIRE(i <= (uint16_t) kj::maxValue);
				
				backend.accept(capnp::DynamicEnum(type.asEnum(), (uint16_t) i));
				return;
			}
			
			if(type.isBool()) {
				if(i == 1)
					backend.accept(true);
				else if(i == 0)
					backend.accept(false);
				else
					KJ_FAIL_REQUIRE("Only 1 and 0 may be converted to bool");
				return;
			}
			
			backend.accept(i);
		}
		
		void acceptNull() override {
			ACCEPT_FWD(acceptNull())
			backend.accept(capnp::Void());
		}
		
		void acceptDouble(double d) override {
			ACCEPT_FWD(acceptDouble(d))
			backend.accept(d);
		}
		
		void acceptInt(int64_t i) override {
			ACCEPT_FWD(acceptInt(i))
			acceptInteger<int64_t>(i);
		}
		
		void acceptUInt(uint64_t i) override {
			ACCEPT_FWD(acceptUInt(i))
			acceptInteger<uint64_t>(i);
		}
		
		void acceptBool(bool b) override {
			ACCEPT_FWD(acceptBool(b))
			backend.accept(b);
		}
		
		void acceptString(kj::StringPtr s) override {
			ACCEPT_FWD(acceptString(s))
			backend.accept(capnp::Text::Reader(s));
			
			auto type = expectedType();
			using ST = capnp::schema::Type;
			
			switch(type.which()) {
				case ST::DATA:
				case ST::ANY_POINTER:
				case ST::LIST:
					KJ_FAIL_REQUIRE("Value type mismatch. Can not cast string as type", type);
				
				case ST::TEXT:
					backend.accept(capnp::Text::Reader(s));
					break;
				
				case ST::VOID:
					backend.accept(capnp::Void());
					break;
				
				case ST::ENUM: {
					auto enumerant = type.asEnum().getEnumerantByName(s);
					backend.accept(DynamicEnum(enumerant));
					break;
				}
				
				case ST::STRUCT: {
					backend.beginObject();
					backend.acceptKey(s);
					backend.accept(capnp::Void());
					backend.finish();
					break;
				}
				
				case ST::FLOAT32:
				case ST::FLOAT64: {
					kj::String asString = kj::heapString(input.as<capnp::Text>());
					for(char& c : asString)
						c = std::tolower(c);
					
					#define HANDLE_VAL(val, result) \
						if(asString == val) { \
							backend.accept(result); \
							break; \
						}
						
					HANDLE_VAL(".nan", std::numeric_limits<double>::quiet_NaN())
					HANDLE_VAL(".inf", std::numeric_limits<double>::infinity())
					HANDLE_VAL("-.inf", -std::numeric_limits<double>::infinity())
					
					#define HANDLE_VAL(size, val, result) \
						if(asString.substr(0, size) == val) { \
							backend.accept(result); \
							break; \
						}
						
					HANDLE_VAL(3, "nan", std::numeric_limits<double>::quiet_NaN())
					HANDLE_VAL(3, "inf", std::numeric_limits<double>::infinity())
					HANDLE_VAL(4, "-inf", -std::numeric_limits<double>::infinity())
					
					#undef HANDLE_VAL
								
					backend.accept(s.parseAs<double>());
					break;
				}
				
				case ST::UINT8:
				case ST::UINT16:
				case ST::UINT32:
				case ST::UINT64: {
					backend.accept(s.parseAs<uint64_t>());
					break;
				}
				
				case ST::INT8:
				case ST::INT16:
				case ST::INT32:
				case ST::INT64: {
					backend.accept(s.parseAs<int64_t>());
					break;
				}
				
				case ST::BOOL: {
					kj::String asString = kj::heapString(s);
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
					HANDLE_VAL("1", true);
					HANDLE_VAL("0", true);
					
					#undef HANDLE_VAL
					
					KJ_FAIL_REQUIRE("Unable to convert value to bool", asString);
				}
			
				case ST::INTERFACE: {
					capnp::Capability::Client brokenCap(KJ_EXCEPTION(FAILED, s));
					backend.accept(brokenCap.castAs<DynamicCapability>(type.asInterface()));
					break;
				}
			}
		}
		
		void acceptData(kj::ArrayPtr<const kj::byte> d) override {
			ACCEPT_FWD(acceptData(d))
			backend.accept(capnp::Data::Reader(d));
		}
		
		bool done() override {
			return backend.isDone;
		}
	
	private:		
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

DynamicValue::Reader Node::asValue() {
	KJ_REQUIRE(!payload.is<MapPayload>() && !payload.is<ListPayload>(), "Can not convert map or list payloads to value");
	
	if(payload.is<kj::String>())
		return capnp::Text::Reader(payload.get<kj::String>());
	
	if(payload.is<Array<byte>>())
		return capnp::Data::Reader(payload.get<Array<byte>>());
	
	if(payload.is<double>())
		return payload.get<double>();
	
	if(payload.is<uint64_t>())
		return payload.get<uint64_t>();
	
	if(payload.is<int64_t>())
		return payload.get<int64_t>();
	
	if(payload.is<bool>())
		return payload.get<bool>();

	if(payload.is<capnp::Void>())
		return capnp::Void();
	
	payload.allHandled<9>();
}

Own<Visitor> createVisitor(DynamicStruct::Builder b) {
	return makeBuilderStack(kj::heap<ExternalStructSink>(b));
}

Own<Visitor> createVisitor(capnp::ListSchema schema, ListInitializer initializer) {
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

void load(kj::BufferedInputStream& is, Node& n, const Dialect&) {
	auto v = createVisitor(n);
	load(is, *v, dialect);
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