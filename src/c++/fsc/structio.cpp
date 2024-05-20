#include "structio.h"

#include <kj/encoding.h>
#include <capnp/serialize-packed.h>

using capnp::DynamicList;
using capnp::DynamicValue;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::StructSchema;

namespace fsc { namespace structio {

namespace {
	void checkList(Maybe<capnp::Type> tp) {
		KJ_IF_MAYBE(pType, tp) {
			if(pType -> isList())
				return;
			
			if(pType -> isAnyPointer()) {
				auto apKind = pType -> whichAnyPointerKind();
				KJ_REQUIRE(apKind == capnp::schema::Type::AnyPointer::Unconstrained::ANY_KIND || apKind == capnp::schema::Type::AnyPointer::Unconstrained::LIST);
				return ;
			}
			
			KJ_FAIL_REQUIRE("Type is not a valid list type");
		}
	}
	
	void checkStruct(Maybe<capnp::Type> tp) {
		KJ_IF_MAYBE(pType, tp) {
			if(pType -> isStruct())
				return;
			
			if(pType -> isAnyPointer()) {
				auto apKind = pType -> whichAnyPointerKind();
				KJ_REQUIRE(apKind == capnp::schema::Type::AnyPointer::Unconstrained::ANY_KIND || apKind == capnp::schema::Type::AnyPointer::Unconstrained::STRUCT);
				return;
			}
			
			KJ_FAIL_REQUIRE("Type is not a valid struct type");
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
		virtual ~Sink() noexcept(false) {};
		
		virtual DynamicStruct::Builder newObject() = 0;
		virtual DynamicList::Builder newList(size_t) = 0;
		virtual void accept(DynamicValue::Reader) = 0;
		virtual capnp::Type expectedType() = 0;
		
		virtual void finish() {};
		
		// virtual void acceptKey(kj::StringPtr) = 0;
		
		// capnp::Orphanage orphanage() = 0;
	};
	
	struct ExternalStructSink : public Sink {
		DynamicStruct::Builder builder;
		bool consumed = false;
		
		ExternalStructSink(DynamicStruct::Builder b) : builder(b) {}
		
		DynamicStruct::Builder newObject() override {
			KJ_REQUIRE(!consumed, "Internal error");
			consumed = true;
			return builder;
		}
		
		DynamicList::Builder newList(size_t) override {
			KJ_FAIL_REQUIRE("Internal error");
		}
		
		void accept(DynamicValue::Reader) override {
			KJ_FAIL_REQUIRE("Internal error");
		}
		
		capnp::Type expectedType() override {
			return builder.getSchema();
		}
		
		/*void acceptKey(kj::StringPtr) override {
			KJ_FAIL_REQUIRE("Internal error");
		}*/
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
		
		DynamicList::Builder newList(size_t s) override {
			KJ_REQUIRE(!consumed, "Internal error");
			consumed = true;
			return initializer(s);
		}
		
		void accept(DynamicValue::Reader) override {
			KJ_FAIL_REQUIRE("Internal error");
		}
		
		capnp::Type expectedType() override {
			return listSchema;
		}
	};
	
	struct StructSink : public Sink {
		DynamicStruct::Builder builder;
		Maybe<capnp::StructSchema::Field> currentField = nullptr;
		Maybe<capnp::StructSchema::Field> previouslySetUnionField = nullptr;
		
		StructSink(DynamicStruct::Builder builder) :
			builder(builder)
		{}
		
		DynamicStruct::Builder newObject() override {
			KJ_IF_MAYBE(pField, currentField) {
				auto field = *pField;
				currentField = nullptr;
				KJ_REQUIRE(field.getType().isStruct(), "Can only decode maps as structs");
				
				return builder.init(field).as<DynamicStruct>();
			}
			KJ_FAIL_REQUIRE("Trying to allocate map entry without key");
		}
		
		DynamicList::Builder newList(size_t size) override {
			KJ_IF_MAYBE(pField, currentField) {
				auto field = *pField;
				currentField = nullptr;
				
				KJ_REQUIRE(field.getType().isList(), "Can only decode arrays as lists");
				return builder.init(field, size).as<DynamicList>();
			}
			KJ_FAIL_REQUIRE("Trying to allocate map entry without key");
		}
		
		void accept(DynamicValue::Reader val) override {
			KJ_IF_MAYBE(pField, currentField) {
				auto field = *pField;
				currentField = nullptr;
				
				if(val.getType() == DynamicValue::VOID)
					builder.clear(field);
				else
					builder.set(field, val);
			} else {
				StructSchema::Field newField;
				
				// val is the field key
				switch(val.getType()) {
					case DynamicValue::INT:
					case DynamicValue::UINT:
						newField = builder.getSchema().getFields()[val.as<unsigned int>()];
						break;
					case DynamicValue::TEXT:
						newField = builder.getSchema().getFieldByName(val.as<capnp::Text>());
						break;
					default:
						KJ_FAIL_REQUIRE("Only integer and text map keys supported");
				}	
			
				// Make sure we don't set union fields twice
				if(newField.getProto().getDiscriminantValue() != capnp::schema::Field::NO_DISCRIMINANT) {
					KJ_IF_MAYBE(pPrevSet, previouslySetUnionField) {
						if(*pPrevSet != newField) {
							kj::StringPtr currentField = newField.getProto().getName();
							kj::StringPtr previousField = pPrevSet -> getProto().getName();
							
							KJ_FAIL_REQUIRE("Can not set two fields of the same union", currentField, previousField);
						}
					}
					previouslySetUnionField = newField;
				}

				currentField = newField;
			}
		}
		
		capnp::Type expectedType() override {
			KJ_IF_MAYBE(pField, currentField) {
				return pField -> getType();
			}
			
			// Key types are text
			return capnp::Type(capnp::schema::Type::TEXT);
		}
		
		/*void acceptKey(kj::StringPtr key) override {
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
		}*/
	};
	
	struct DataRefSink : public Sink {
		enum KeyState { NO_KEY, UNKNOWN_KEY, DATA_KEY };
		KeyState keyState = NO_KEY;
		
		capnp::MallocMessageBuilder message;
		
		capnp::InterfaceSchema refSchema;
		capnp::Type valueType;
		Sink& parentSink;
		
		DataRefSink(capnp::InterfaceSchema newRefType, Sink& parent) :
			refSchema(newRefType), valueType(extractValueType(refSchema)), parentSink(parent)
		{
			KJ_REQUIRE(valueType.isData() || valueType.isStruct(), "Only Data and Struct types supported for DataRef");
		}
		
		capnp::Type expectedType() override {
			switch(keyState) {
				case NO_KEY: return capnp::Type(capnp::schema::Type::TEXT);
				case UNKNOWN_KEY: return capnp::Type(capnp::schema::Type::VOID);
				case DATA_KEY: return valueType;
			}
			KJ_FAIL_REQUIRE("Invalid key state");
		}
		
		DynamicStruct::Builder newObject() override {
			KJ_REQUIRE(keyState == DATA_KEY, "Internal error");
			KJ_REQUIRE(valueType.isStruct(), "Can only load structured data into struct targets");
			keyState = NO_KEY;
			
			return message.initRoot<capnp::DynamicStruct>(valueType.asStruct());
		}
		
		DynamicList::Builder newList(size_t size) override {
			KJ_UNIMPLEMENTED("List refs not supported");
			/*KJ_REQUIRE(keyState == DATA_KEY, "Internal error");
			KJ_REQUIRE(valueType.isList(), "Can only load list data into list targets");
			
			return message.initRoot<capnp::DynamicList>(valueType.asList(), size);*/
		}
		
		void accept(DynamicValue::Reader val) override {
			switch(keyState) {
				case DATA_KEY:
					KJ_REQUIRE(valueType.isData(), "Target type must be binary");
					KJ_REQUIRE(val.getType() == DynamicValue::DATA, "Source data must be a binary literal");
					
					message.setRoot(val.as<capnp::Data>());
					keyState = NO_KEY;
					break;
				case UNKNOWN_KEY:
					keyState = NO_KEY;
					break;
				
				case NO_KEY:
					if(val.getType() == DynamicValue::TEXT) {
						auto textReader = val.as<capnp::Text>();
						if(textReader == "data") {
							keyState = DATA_KEY;
							break;
						}
					}
					keyState = UNKNOWN_KEY;
			}
		}
		
		void finish() override {
			auto& ds = getActiveThread().dataService();
			
			capnp::Capability::Client untyped = nullptr;
			
			if(valueType.isData()) {
				untyped = ds.publish(message.getRoot<capnp::Data>());
			} else if(valueType.isStruct()) {
				untyped = ds.publish(message.getRoot<capnp::DynamicStruct>(valueType.asStruct()).asReader());
			} else {
				KJ_FAIL_REQUIRE("Internal error");
			}
			
			auto typed = untyped.castAs<capnp::DynamicCapability>(refSchema);
			parentSink.accept(typed);
		}
	private:
		static capnp::Type extractValueType(capnp::InterfaceSchema refSchema) {
			constexpr uint64_t DR_ID = capnp::typeId<DataRef<capnp::AnyPointer>>();
			
			KJ_REQUIRE(
				refSchema.getProto().getId() == DR_ID,
				"Type must be a DataRef instance"
			);
			
			return refSchema.getBrandArgumentsAtScope(DR_ID)[0];
		}
	};
	
	struct ListSink : public Sink {
		DynamicList::Builder dst;
		size_t offset = 0;
		
		ListSink(DynamicList::Builder ndst) :
				dst(ndst)
		{}
		
		DynamicStruct::Builder newObject() override {
			return dst[offset++].as<DynamicStruct>();
		}
		
		DynamicList::Builder newList(size_t s) override {
			return dst.init(offset++, s).as<DynamicList>();
		}
		
		void accept(DynamicValue::Reader val) override {		
			if(val.getType() == DynamicValue::VOID) {
				// Since these lists are freshly initialized, we can just skip this value
				++offset;
				return;
			}
			dst.set(offset++, val);
		}
		
		capnp::Type expectedType() override {
			return dst.getSchema().getElementType();
		}
		
		/*void acceptKey(kj::StringPtr) override {
			KJ_FAIL_REQUIRE("Can not assign keys to lists, only in objects");
		}
		
		capnp::Orphanage orphanage() override {
			return capnp::Orphanage::getForMessageContaining(dst);
		}*/
	};
	
	struct BuilderStack {
		kj::Vector<Own<Sink>> stack;
		bool isDone = false;
		
		BuilderStack(Own<Sink>&& first) {
			stack.add(mv(first));
		}
		
		capnp::Type expectedType() {
			KJ_REQUIRE(!stack.empty());
			
			return stack.back() -> expectedType();
		}
		
		void beginObject() {
			KJ_REQUIRE(!isDone);
			
			// If the destination type is a DataRef, we add a different type of object sink
			// that passes through to the parent object. Cases where we are a non-DR interface
			// get caught by the value converter.
			auto type = stack.back() -> expectedType();
			if(type.isInterface()) {
				stack.add(kj::heap<DataRefSink>(type.asInterface(), *stack.back()));
				return;
			}
			
			stack.add(kj::heap<StructSink>(stack.back() -> newObject()));
		}
		
		void beginArray(size_t size) {			
			KJ_REQUIRE(!isDone);
			
			stack.add(kj::heap<ListSink>(stack.back() -> newList(size)));
		}
		
		void accept(capnp::DynamicValue::Reader input) {
			KJ_REQUIRE(!isDone);
			stack.back() -> accept(input);
			
			if(stack.size() == 1)
				isDone = true;
		}
		
		void finish() {	
			KJ_REQUIRE(!isDone);
			
			stack.back() -> finish();
			stack.removeLast();
			
			if(stack.size() <= 1)
				isDone = true;
		}
	};
	
	/* This class implements multiple important conversions:
	   - Contains ways of constructing most value types from texts
	   - Supports the shorthand notation "field" for {"field" : null} on structs
	   - Filters out objects for non-dataref capabilities
	*/
	struct ValueConverter : public Visitor {
		BuilderStack backend;
		size_t ignoreDepth = 0;
		bool replaying = false;
		
		struct Tape {
			Node node;
			Own<Visitor> recorder;
			
			Tape() : recorder(createVisitor(node)) {}
		};
		Maybe<Tape> tape;
		
		ValueConverter(Own<Sink>&& firstSink) :
			backend(mv(firstSink))
		{
			this -> supportsIntegerKeys = true;
		}
		
		#define ACCEPT_FWD(expr) \
			KJ_IF_MAYBE(pTape, tape) { if(!replaying) { \
				pTape -> recorder -> expr; \
				\
				if(pTape -> recorder -> done()) { \
					replaying = true; \
					save(mv(pTape -> node), *this); \
					replaying = false; \
					tape = nullptr; \
				} \
				return; \
			}}
		
		void beginObject(Maybe<size_t> s) override {
			ACCEPT_FWD(beginObject(s))
			
			if(ignoreDepth > 0 || ignoreBegin()) {
				++ignoreDepth;
				return;
			}
			
			backend.beginObject();
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
				KJ_REQUIRE(tape == nullptr, "Internal error");
				
				Tape& newTape = tape.emplace();
				newTape.recorder -> beginArray(nullptr);
			}
		}
		
		void endArray() override {
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
			
			if(type.isStruct()) {
				backend.beginObject();
				backend.accept(i);
				backend.accept(capnp::Void());
				backend.finish();
				return;
			}
			
			backend.accept(i);
		}
		
		void acceptNull() override {
			ACCEPT_FWD(acceptNull())
			if(ignoreDepth > 0)
				return;
			backend.accept(capnp::Void());
		}
		
		void acceptDouble(double d) override {
			ACCEPT_FWD(acceptDouble(d))
			if(ignoreDepth > 0)
				return;
			backend.accept(d);
		}
		
		void acceptInt(int64_t i) override {
			ACCEPT_FWD(acceptInt(i))
			if(ignoreDepth > 0)
				return;
			acceptInteger<int64_t>(i);
		}
		
		void acceptUInt(uint64_t i) override {
			ACCEPT_FWD(acceptUInt(i))
			if(ignoreDepth > 0)
				return;
			acceptInteger<uint64_t>(i);
		}
		
		void acceptBool(bool b) override {
			ACCEPT_FWD(acceptBool(b))
			if(ignoreDepth > 0)
				return;
			backend.accept(b);
		}
		
		void acceptString(kj::StringPtr s) override {
			ACCEPT_FWD(acceptString(s))
			
			if(ignoreDepth > 0)
				return;
			
			auto type = backend.expectedType();
			using ST = capnp::schema::Type;
			
			switch(type.which()) {
				case ST::DATA: {
					auto decoded = kj::decodeBase64(s);					
					backend.accept(capnp::Data::Reader(decoded));
					return;
				}
					
				case ST::ANY_POINTER: {
					auto decoded = kj::decodeBase64(s);
					kj::ArrayInputStream inputStream(decoded);

					capnp::PackedMessageReader messageReader(inputStream);
					backend.accept(messageReader.getRoot<capnp::AnyPointer>());
					return;
				}
					
				case ST::LIST:
					KJ_FAIL_REQUIRE("Value type mismatch. Can not cast string as list type");
				
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
					backend.accept(capnp::Text::Reader(s));
					backend.accept(capnp::Void());
					backend.finish();
					break;
				}
				
				case ST::FLOAT32:
				case ST::FLOAT64: {
					kj::String asString = kj::heapString(s);
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
					
					#undef HANDLE_VAL
					
					#define HANDLE_VAL(size, val, result) \
						if(asString.slice(0, size) == kj::StringPtr(val)) { \
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
			
			if(ignoreDepth > 0)
				return;
			
			auto type = backend.expectedType();
			using ST = capnp::schema::Type;
			
			switch(type.which()) {
				case ST::VOID:
					backend.accept(capnp::Void());
					break;
					
				case ST::DATA:
					backend.accept(capnp::Data::Reader(d));
					return;
					
				case ST::ANY_POINTER: {			
					// Check alignment
					kj::ArrayInputStream inputStream(d);

					capnp::PackedMessageReader messageReader(inputStream);
					backend.accept(messageReader.getRoot<capnp::AnyPointer>());
					return;
				}
					
				case ST::LIST:
					KJ_FAIL_REQUIRE("Can not read raw data as list");
				case ST::TEXT:	
					KJ_FAIL_REQUIRE("Can not read raw data as text");		
				case ST::ENUM:
					KJ_FAIL_REQUIRE("Can not read raw data as enum");
				case ST::STRUCT:
					KJ_FAIL_REQUIRE("Can not read raw data as struct");
				case ST::FLOAT32:
					KJ_FAIL_REQUIRE("Can not read raw data as float32");
				case ST::FLOAT64:
					KJ_FAIL_REQUIRE("Can not read raw data as float64");				
				case ST::UINT8:
					KJ_FAIL_REQUIRE("Can not read raw data as uint8");
				case ST::UINT16:
					KJ_FAIL_REQUIRE("Can not read raw data as uint16");
				case ST::UINT32:
					KJ_FAIL_REQUIRE("Can not read raw data as uint32");
				case ST::UINT64:
					KJ_FAIL_REQUIRE("Can not read raw data as uint64");				
				case ST::INT8:
					KJ_FAIL_REQUIRE("Can not read raw data as int8");
				case ST::INT16:
					KJ_FAIL_REQUIRE("Can not read raw data as int16");
				case ST::INT32:
					KJ_FAIL_REQUIRE("Can not read raw data as int32");
				case ST::INT64:
					KJ_FAIL_REQUIRE("Can not read raw data as int64");
				case ST::BOOL:
					KJ_FAIL_REQUIRE("Can not read raw data as bool");
				case ST::INTERFACE:
					KJ_FAIL_REQUIRE("Can not read raw data as capability");
			}
		}
		
		bool done() override {
			return backend.isDone;
		}
	
	private:		
		bool ignoreBegin() {
			auto tp = backend.expectedType();
			
			if(tp.isVoid() || tp.isAnyPointer()) {
				backend.accept(capnp::Void());
				return true;
			} else if(tp.isInterface()) {
				// DataRef targets have beginObject passed through (they get redirected to a special sink)
				constexpr uint64_t DR_ID = capnp::typeId<DataRef<capnp::AnyPointer>>();
				if(tp.asInterface().getProto().getId() == DR_ID)
					return false;
				
				capnp::Capability::Client brokenCap(KJ_EXCEPTION(FAILED, "Can not restore capability"));
				backend.accept(brokenCap.castAs<DynamicCapability>(tp.asInterface()));
				return true;
			}
			
			return false;
		}
	};
	
	void saveValue(DynamicValue::Reader, Visitor&, const SaveOptions&, Maybe<kj::WaitScope&> ws);
	void saveStruct(DynamicStruct::Reader, Visitor&, const SaveOptions&, Maybe<kj::WaitScope&> ws);
	void saveList(DynamicList::Reader, Visitor&, const SaveOptions&, Maybe<kj::WaitScope&> ws);
	
	struct DefaultCapabilityStrategy : public SaveOptions::CapabilityStrategy {
		void saveCapability(capnp::DynamicCapability::Client ref, Visitor& v, const SaveOptions& opts, Maybe<kj::WaitScope&> maybeWS) const override {
			using GenericRef = DataRef<capnp::AnyPointer>;
			
			KJ_IF_MAYBE(pWS, maybeWS) {
				kj::WaitScope& ws = *pWS;
				
				// Check whether we actually are a DataRef
				auto schema = ref.getSchema();
				
				constexpr uint64_t DR_ID = capnp::typeId<GenericRef>();
				if(schema.getProto().getId() != DR_ID)
					goto writeDefault;
				
				auto asRef = capnp::Capability::Client(ref).castAs<DataRef<capnp::AnyPointer>>();
				auto localRef = getActiveThread().dataService().download(asRef).wait(ws);
				
				v.beginObject(2);
				
				v.acceptString("metadata");
				save(localRef.getMetadata(), v, opts, maybeWS);
				
				v.acceptString("data");
				
				auto payloadType = schema.getBrandArgumentsAtScope(DR_ID)[0];
				if(payloadType.isData()) {
					KJ_REQUIRE(localRef.getFormat().isRaw(), "Data-typed messages not supported");
					
					capnp::Data::Reader rawData = localRef.getRaw();
					save(rawData, v, opts, ws);
				} else {
					KJ_REQUIRE(payloadType.isStruct(), "Only data and struct refs supported");
					KJ_REQUIRE(localRef.getFormat().isSchema(), "Type mismatch between ref and expected data");
					
					capnp::AnyPointer::Reader untyped = localRef.get();
					DynamicStruct::Reader typed = untyped.getAs<DynamicStruct>(payloadType.asStruct());
					
					save(typed, v, opts, ws);
				}
				
				v.endObject();
				
				return;
			}
			
			writeDefault:
				v.acceptString("<capability>");
		}
	};
	
	static DefaultCapabilityStrategy DEFAULT_CAP_INSTANCE;
	
	void saveValue(DynamicValue::Reader in, Visitor& v, const SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
		switch(in.getType()) {
			case DynamicValue::STRUCT:
				saveStruct(in.as<DynamicStruct>(), v, opts, ws);
				break;
			case DynamicValue::LIST:
				try {
					auto asList = in.as<DynamicList>();
					if(asList.size() > 0)
						auto e = asList[0];					
				} catch(kj::Exception& e) {
					KJ_LOG(WARNING, "Failed to obtain list, emitting null instead.");
					v.acceptNull();
					break;
				}
				
				saveList(in.as<DynamicList>(), v, opts, ws);
				break;
			case DynamicValue::CAPABILITY: {
				auto asCap = in.as<DynamicCapability>();
				auto hook = capnp::ClientHook::from(kj::cp(asCap));
				if(hook -> isNull()) {
					v.acceptNull();
				} else {
					opts.capabilityStrategy -> saveCapability(asCap, v, opts, ws);
				}
				break;
			}
			case DynamicValue::ENUM:
				if(!opts.integerKeys) {
					KJ_IF_MAYBE(pEnumerant, in.as<DynamicEnum>().getEnumerant()) {
						v.acceptString(pEnumerant -> getProto().getName());
						break;
					}
				}
				
				v.acceptUInt(in.as<DynamicEnum>().getRaw());
				break;
			case DynamicValue::FLOAT:
				v.acceptDouble(in.as<double>());
				break;
			case DynamicValue::UINT:
				v.acceptUInt(in.as<uint64_t>());
				break;
			case DynamicValue::INT:
				v.acceptInt(in.as<int64_t>());
				break;
			case DynamicValue::BOOL:
				v.acceptBool(in.as<bool>());
				break;
			case DynamicValue::VOID:
				v.acceptNull();
				break;
			case DynamicValue::TEXT:
				v.acceptString(in.as<capnp::Text>());
				break;
			case DynamicValue::DATA:
				v.acceptData(in.as<capnp::Data>());
				break;
			case DynamicValue::ANY_POINTER: {
				capnp::MallocMessageBuilder tmpMessage;
				tmpMessage.setRoot(in.as<capnp::AnyPointer>());
				
				kj::VectorOutputStream vos;
				capnp::writePackedMessage(vos, tmpMessage);
				v.acceptData(vos.getArray());
				break;
			}
			case DynamicValue::UNKNOWN:
				v.acceptString("<unknown>");
				break;
		}
	}
	
	void saveStruct(DynamicStruct::Reader in, Visitor& v, const SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
		// If we have no non-union fields and the active union field is a
		// default-valued one, then we write only the field name (reads nicer
		
		if(in.getSchema().getNonUnionFields().size() == 0 && opts.compact) {
			auto maybeActive = in.which();
			KJ_IF_MAYBE(pActive, maybeActive) {
				auto& active = *pActive;
				if(!in.has(active, capnp::HasMode::NON_DEFAULT)) {
					if(opts.integerKeys) {
						v.acceptUInt(active.getIndex());
					} else {
						v.acceptString(active.getProto().getName());
					}
					return;
				}
			}
		}
		
		auto shouldEmitField = [&](capnp::StructSchema::Field field) {
			// Check if field is inactive union field
			if(field.getProto().getDiscriminantValue() != capnp::schema::Field::NO_DISCRIMINANT) {
				KJ_IF_MAYBE(pField, in.which()) {
					// Active union fields ALWAYS HAVE to be emitted
					return *pField == field;
				}
				return false;
			} 
			
			// No point in emitting void non-union fields
			if(field.getType().isVoid())
				return false;
			
			// Don't emit null-valued fields for capabilities & AnyPointer
			if(field.getType().isAnyPointer() || field.getType().isInterface()) {
				if(!in.has(field, capnp::HasMode::NON_DEFAULT))
					return false;
			}
			
			// In compact representation we can omit default-valued fields of any kind
			if(opts.compact && !in.has(field, capnp::HasMode::NON_DEFAULT)) {
				return false;
			}
			
			return true;
		};
		
		size_t fieldCount = 0;
		for(auto field : in.getSchema().getFields()) {
			if(shouldEmitField(field))
				++fieldCount;
		}
		
		v.beginObject(fieldCount);
		
		auto emitField = [&](capnp::StructSchema::Field field) {
			auto type = field.getType();
			
			if(opts.integerKeys && v.supportsIntegerKeys)
				v.acceptUInt(field.getIndex());
			else
				v.acceptString(field.getProto().getName());
			
			// We CAN NOT emit nullptr structs, since this can cause
			// infinite depth on recursive structures.
			if(field.getType().isStruct() && !in.has(field)) {
				v.acceptNull();
				return;
			}
			
			saveValue(in.get(field), v, opts, ws);
		};
		
		for(auto field : in.getSchema().getFields()) {
			if(shouldEmitField(field))
				emitField(field);
		}
		
		v.endObject();
	}
	
	void saveList(DynamicList::Reader list, Visitor& v, const SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
		v.beginArray(list.size());
		
		for(auto el : list)
			saveValue(el, v, opts, ws);
		
		v.endArray();
	}
	
	Own<Visitor> makeBuilderStack(Own<Sink>&& sink) {
		return kj::heap<ValueConverter>(mv(sink));
	}
	
	struct DebugVisitor : public Visitor {		
		virtual void beginObject(Maybe<size_t> s) {
			KJ_IF_MAYBE(pSize, s) {
				KJ_DBG("beginObject(s)", *pSize);
			} else {
				KJ_DBG("beginObject(nullptr)");
			}
		}
		virtual void endObject() {
			KJ_DBG("endObject()");
		}
		
		virtual void beginArray(Maybe<size_t> s) {
			KJ_IF_MAYBE(pSize, s) {
				KJ_DBG("beginArray(s)", *pSize);
			} else {
				KJ_DBG("beginArray(nullptr)");
			}
		}
		virtual void endArray() {
			KJ_DBG("endArray()");
		}
		
		virtual void acceptNull() {
			KJ_DBG("acceptNull()");
		}
		virtual void acceptDouble(double d) {
			KJ_DBG("acceptDouble(d)", d);
		}
		virtual void acceptInt(int64_t i) {
			KJ_DBG("acceptInt(i)", i);
		}
		virtual void acceptUInt(uint64_t u) {
			KJ_DBG("acceptUInt(u)", u);
		}
		virtual void acceptString(kj::StringPtr s) {
			KJ_DBG("acceptString(s)", s);
		}
		virtual void acceptData(ArrayPtr<const byte> b) {
			KJ_DBG("acceptData(b)", b);
		}
		virtual void acceptBool(bool b) {
			KJ_DBG("acceptBool(b)", b);
		}
		
		virtual bool done() { return false; }
		
		//! Whether this visitor allows integer map keys
		bool supportsIntegerKeys = true;
	};
}

SaveOptions::CapabilityStrategy* const SaveOptions::CapabilityStrategy::DEFAULT = &DEFAULT_CAP_INSTANCE;

Own<Visitor> createVisitor(DynamicStruct::Builder b) {
	return makeBuilderStack(kj::heap<ExternalStructSink>(b));
}

Own<Visitor> createVisitor(capnp::ListSchema schema, ListInitializer initializer) {
	return makeBuilderStack(kj::heap<ExternalListSink>(schema, mv(initializer)));
}

Own<Visitor> createDebugVisitor() {
	return kj::heap<DebugVisitor>();
}

Own<Visitor> createVisitor(kj::BufferedOutputStream& os, const Dialect& dialect, const SaveOptions& opts) {
	if(dialect.language == Dialect::YAML) {
		return internal::createYamlcppWriter(os, dialect);
	} else {
		return internal::createJsonconsWriter(os, dialect);
	}
}

void load(kj::ArrayPtr<const kj::byte> buf, Visitor& v, const Dialect& d) {
	kj::ArrayInputStream is(buf);
	load(is, v, d);
}

void load(kj::BufferedInputStream& is, Visitor& visitor, const Dialect& dialect) {
	if(dialect.language == Dialect::YAML) {
		internal::yamlcppLoad(is, visitor, dialect);
	} else {
		internal::jsonconsLoad(is, visitor, dialect);
	}
}

void save(DynamicValue::Reader reader, Visitor& v, const SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
	saveValue(reader, v, opts, ws);
}

void save(DynamicValue::Reader reader, kj::BufferedOutputStream& os, const Dialect& dialect, const SaveOptions& opts, Maybe<kj::WaitScope&> ws) {	
	saveValue(reader, *createVisitor(os, dialect, opts), opts, ws);
}

Array<kj::byte> saveToArray(DynamicValue::Reader reader, const Dialect& dialect, const SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
	kj::VectorOutputStream os;
	save(reader, os, dialect, opts, ws);
	
	return kj::heapArray(os.getArray().asBytes());
}

kj::String saveToString(DynamicValue::Reader reader, const Dialect& dialect, const SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
	kj::VectorOutputStream os;
	save(reader, os, dialect, opts, ws);
	
	const char zero = 0;
	os.write(&zero, 1);
	
	return kj::heapString(os.getArray().asChars());
}

}}