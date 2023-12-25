#include "json.h"

#include <jsoncons/json_encoder.hpp>
#include <jsoncons/json_cursor.hpp>

#include <jsoncons_ext/cbor/cbor_encoder.hpp>
#include <jsoncons_ext/cbor/cbor_cursor.hpp>

#include <jsoncons_ext/bson/bson_encoder.hpp>
#include <jsoncons_ext/bson/bson_cursor.hpp>

#include <kj/function.h>

using capnp::DynamicList;
using capnp::DynamicValue;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;

// This file is based on yaml.cpp, but adapted to use the goldfish CBOR / JSON writer.

namespace fsc {

namespace {
	
	template<typename T>
	struct InputSource {
		static_assert(sizeof(T) == 1, "T must be a char type");
		
		using value_type = T;
		
		struct PeekResult {
			T value;
			bool eof;
		};
		
		InputSource(kj::BufferedInputStream& s) : stream(s) {}
		
		bool eof() const {
			return eofFlag;
		}
		
		bool is_error() {
			return false;
		}
		
		size_t position() const {
			return streamPosition;
		}
		
		void ignore(size_t skip) {
			if(skip <= buffer.size()) {
				buffer = buffer.slice(skip, buffer.size());
			} else {
				buffer = nullptr;
			}
			
			consumedFromBuffer += skip;
			streamPosition += skip;
		}
		
		PeekResult peek() {
			fill();
			
			if(eof())
				return PeekResult{0, true};
			
			return PeekResult{buffer[0], false};
		}
			
		jsoncons::span<const T> read_buffer() {
			fill();
			
			jsoncons::span<const T> result(
				reinterpret_cast<const T*>(buffer.begin()), buffer.size()
			);
			
			streamPosition += buffer.size();
			
			consumedFromBuffer += buffer.size();
			buffer = nullptr;
			
			return result;
		}
		
		size_t read(T* dst, size_t length) {
			if(length < buffer.size()) {
				// Try to serve read from buffer
				memcpy(dst, buffer.begin(), length);
				
				buffer = buffer.slice(length, buffer.size());
				consumedFromBuffer += length;
				
				streamPosition += length;
				
				return length;
			} else {
				// Indicate skipped bytes
				stream.skip(consumedFromBuffer);
				consumedFromBuffer = 0;
				
				buffer = nullptr;
				
				size_t bytesRead = stream.tryRead(dst, 1, length);
				streamPosition += bytesRead;
				
				return bytesRead;
			}
		}
			
		
	private:
		void fill() {
			if(buffer == nullptr) {
				stream.skip(consumedFromBuffer);
				consumedFromBuffer = 0;
				
				buffer = stream.tryGetReadBuffer();
			}
			
			if(buffer == nullptr)
				eofFlag = true;
		}
		
		kj::BufferedInputStream& stream;
		ArrayPtr<const kj::byte> buffer = nullptr;
		
		size_t streamPosition = 0;
		size_t consumedFromBuffer = 0; // This can be bigger than buffer size
		bool eofFlag = false;
	};
	
	template<typename T>
	struct OutputSink {
		static_assert(sizeof(T) == 1, "T must be a char type");
		
		using value_type = T;
		
		OutputSink(kj::BufferedOutputStream& s) : stream(s) {}
		
		void flush() {
			stream.write(buffer.begin(), bytesUsed);
			buffer = stream.getWriteBuffer();
			bytesUsed = 0;
		}
		
		void push_back(T b) {
			if(bytesUsed >= buffer.size()) {
				flush();
			}
			
			KJ_ASSERT(buffer.size() > 0);
			buffer[bytesUsed++] = static_cast<kj::byte>(b);
		}
		
		void append(const T* start, size_t length) {
			if(bytesUsed + length > buffer.size()) {
				kj::FixedArray<kj::ArrayPtr<const kj::byte>, 2> pieces;
				pieces[0] = buffer.slice(0, bytesUsed);
				pieces[1] = ArrayPtr<const kj::byte>(reinterpret_cast<const kj::byte*>(start), length);
				
				stream.write(pieces);
				
				buffer = stream.getWriteBuffer();
				bytesUsed = 0;
			} else {
				memcpy(buffer.begin() + bytesUsed, start, length);
				bytesUsed += length;
			}
		}
		
	private:
		kj::BufferedOutputStream& stream;
		
		ArrayPtr<kj::byte> buffer = nullptr;
		size_t bytesUsed = 0;
	};
	
	using CborSource = InputSource<uint8_t>;
	using CborSink = OutputSink<uint8_t>;
	
	using JsonSource = InputSource<char>;
	using JsonSink = OutputSink<char>;
	
	using EventType = jsoncons::staj_event_type;
	using Cursor = jsoncons::basic_staj_cursor<char>;
	using Event = const jsoncons::basic_staj_event<char>;
	
	using Encoder = jsoncons::basic_json_visitor<char>;
	
	DynamicValue::Reader parsePrimitive(capnp::Type type, Event& event) {
		using ST = capnp::schema::Type;
		
		switch(type.which()) {
			case ST::ENUM:
				KJ_IF_MAYBE(pEnumerant, type.asEnum().findEnumerantByName(event.get<std::string>())) {
					return DynamicEnum(*pEnumerant);
				} else {
					return DynamicEnum(type.asEnum(), event.get<uint16_t>());
				}
			
			case ST::DATA:
			case ST::TEXT:
			case ST::STRUCT:
			case ST::LIST:
				KJ_FAIL_REQUIRE("parsePrimitive may not be used to decode pointer types");
			
			case ST::VOID:
				return capnp::Void();
			
			case ST::BOOL:
				return event.get<bool>();
			
			case ST::ANY_POINTER:
			case ST::INTERFACE:
				return nullptr;
			
			case ST::FLOAT32:
			case ST::FLOAT64:
				// Custom parsing for String values
				if(event.event_type() == EventType::string_value) {
					std::string asString = event.get<std::string>();
					for(char& c : asString)
						c = std::tolower(c);
					
					if(asString == ".nan")
						return std::numeric_limits<double>::quiet_NaN();
					if(asString == ".inf")
						return std::numeric_limits<double>::infinity();
					if(asString == "-.inf")
						return -std::numeric_limits<double>::infinity();
					
					if(asString.substr(0, 3) == "nan")
						return std::numeric_limits<double>::quiet_NaN();
					if(asString.substr(0, 3) == "inf")
						return std::numeric_limits<double>::infinity();
					if(asString.substr(0, 4) == "-inf")
						return -std::numeric_limits<double>::infinity();
								
					return std::stod(asString);
				} else {
					return event.get<double>();
				}
			
			case ST::UINT8:
			case ST::UINT16:
			case ST::UINT32:
			case ST::UINT64:
				return event.get<uint64_t>();
			
			case ST::INT8:
			case ST::INT16:
			case ST::INT32:
			case ST::INT64:
				return event.get<int64_t>();
		}
		
		KJ_FAIL_REQUIRE("Unknown target primitive type");
	}	
	
	void emitPrimitive(DynamicValue::Reader val, Encoder& encoder) {
		auto type = val.getType();
				
		KJ_REQUIRE(type != DynamicValue::STRUCT);
		KJ_REQUIRE(type != DynamicValue::LIST);
		
		switch(type) {
			case DynamicValue::CAPABILITY: {
				auto asCap = val.as<DynamicCapability>();
				auto hook = capnp::ClientHook::from(kj::mv(asCap));
				if(hook -> isNull())
					encoder.null_value();
				else
					encoder.string_value("<capability>");
				
				break;
			}
			case DynamicValue::ANY_POINTER:
				encoder.string_value("<unknown>");
				break;
				
			case DynamicValue::VOID:
				encoder.null_value();
				break;
				
			case DynamicValue::DATA: {
				auto asBin = val.as<capnp::Data>();
				encoder.byte_string_value(jsoncons::byte_string_view(asBin.begin(), asBin.size()));
				break;
			}
			
			case DynamicValue::TEXT: {
				auto asText = val.as<capnp::Data>();
				encoder.string_value(jsoncons::string_view((const char*) asText.begin(), asText.size()));
				break;
			}
				
			case DynamicValue::ENUM: {
				auto enumerant = val.as<DynamicEnum>().getEnumerant();
				KJ_IF_MAYBE(pEn, enumerant) {
					encoder.string_value(pEn -> getProto().getName().cStr());
				} else {
					encoder.half_value(val.as<DynamicEnum>().getRaw());
				}
				break;
			}
			
			case DynamicValue::BOOL:
				encoder.bool_value(val.as<bool>());
				break;
			
			case DynamicValue::FLOAT: {
				encoder.double_value(val.as<double>());
				break;
			}
			
			case DynamicValue::INT:
				encoder.int64_value(val.as<int64_t>());
				break;
			
			case DynamicValue::UINT:
				encoder.uint64_value(val.as<uint64_t>());
				break;
			
			case DynamicValue::UNKNOWN:
				KJ_FAIL_REQUIRE("Internal error: Unhandled type");
		}
	}
	
	struct Sink {
		using Initializer = kj::Function<capnp::DynamicList::Builder(size_t)>;
		virtual ~Sink() {};
		
		virtual DynamicStruct::Builder newObject() = 0;
		virtual Initializer newList() = 0;
		virtual void accept(Orphan<DynamicValue>) = 0;
		virtual capnp::Type expectedType() = 0;
		virtual void finish() = 0;
		
		virtual void acceptKey(kj::StringPtr) = 0;
		
		
		capnp::Orphanage orphanage() = 0;
	};
	
	struct StructSink : public Sink {
		DynamicStruct::Builder builder;
		Maybe<capnp::StructSchema::Field> currentField = nullptr;
		
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
		Initializer newList() override {
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
			currentField = builder.getFieldByName(key);
		}
		
		void beginList(Maybe<size_t>) override {
			KJ_FAIL_REQUIRE("Can not assign a list to a struct slot");
		}
		
		void beginObject() override {}
		void finish() override {}
		
		capnp::Orphanage orphanage() override {
			return capnp::Orphanage::getForMessageContaining(builder);
		}
	};
	
	struct ListSink {
		Initializer initializer;
		capnp::ListSchema listSchema;
		
		capnp::MallocMessageBuilder tmpMessage;
		
		using TmpList = capnp::Orphan<capnp::DynamicList>;
		kj::Vector<TmpList> finishedLists;
		TmpList currentList;
		size_t currentListComplete = 0;
		
		ListSink(capnp::ListSchema listSchema, ListSink::Initializer&& initializer, Maybe<size_t> maybeSize) :
				initializer(mv(initializer))
				listSchema(listSchema))
		{
			auto elType = listSchema.getElementType();
			size_t size = elType.isBool() ? 64 : 8;
			KJ_IF_MAYBE(pSize, maybeSize) {
				size = *pSize;
			}
			
			currentList = tmpMessage.getOrphanage().newOrphan(listSchema, size);
		}
		
		void finish() override {
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
			
			auto current = currentList.getReader();
			for(auto i : kj::range(0, currentListComplete)) {
				builder.set(outIdx++, current[i]);
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
		Initializer newList() override {
			grow();
			return [dst = currentList.get(), offset = currentListComplete++](size_t size) {
				return dst.init(offset, size);
			};
		}
		
		void accept(DynamicValue::Builder val) override {
			grow();
			auto dst = currentList.get();
			dst.set(currentListComplete++, val);
		}
		
		capnp::Type expectedType() override {
			return listSchema.getElementType();
		}
		
		void acceptKey(kj::StringPtr) override {
			KJ_FAIL_REQUIRE("Can not assign keys to lists, only in objects");
		}
		
		capnp::Orphanage orphanage() override {
			return tmpMessage.getOrphanage();
		}
		
	private:
		void grow() {
			size_t currentSize = currentList.getReader().size();
			if(currentListComplete >= currentSize) {
				finishedLists.append(mv(currentList));
				currentList = tmpMessage.getOrphanage().newOrphan(listSchema, 2 * currentSize);
				currentListComplete = 0;
			}
		}
	};
	
	struct BuilderStack {
		kj::Vector<Own<Sink>> stack;
		size_t ignoreDepth = 0;
		
		CapnpVisitor(capnp::ListSchema schema, ListSink::Initializer&& initializer) {
			stack.append(kj::heap<ListSink>(schema, mv(initializer)));
		}
		
		CapnpVisitor(DynamicStruct::Builder structBuilder) {
			stack.append(kj::heap<StructSink>(structBuilder));
		}
		
		capnp::Type expectedType() {
			KJ_REQUIRE(!stack.empty());
			
			return stack.back().expectedType();
		}
		
		void beginObject() {
			if(ignoreDepth > 0) {
				++ignoreDepth;
				return;
			}
			
			KJ_REQUIRE(!stack.empty());
			
			auto elType = stack.back().expectedType();
			if(elType.isInterface() || elType.isAnyPointer()) {
				beginNull();
				return;
			}
			KJ_REQUIRE(elType.isStruct(), "Trying to assign a map to a non-struct destination");
			
			stack.append(kj::heap<StructSink>(stack.back().newObject()));
		}
		
		void endObject() {
			if(ignoreDepth > 0) {
				--ignoreDepth;
				return;
			}
			
			KJ_REQUIRE(!stack.empty());
			stack.back().finish();
			stack.removeLast();
		}
		
		void beginArray(Maybe<size_t> size = nullptr) {
			if(ignoreDepth > 0) {
				++ignoreDepth;
				return;
			}
			
			KJ_REQUIRE(!stack.empty());
			
			auto elType = stack.back().expectedType();
			if(elType.isInterface() || elType.isAnyPointer()) {
				beginNull();
				return;
			}
			KJ_REQUIRE(elType.isList(), "Trying to assign a list to a non-list destination");
			
			stack.append(kj::heap<ListSink>(elType.asList(), stack.back().newList(), size));
		}
		
		void endArray() {
			if(ignoreDepth > 0) {
				--ignoreDepth;
				return;
			}
			
			KJ_REQUIRE(!stack.empty());
			stack.back().finish();
			stack.removeLast();
		}
		
		void accept(capnp::DynamicValue::Reader input) {
			KJ_REQUIRE(!stack.empty());			
			auto& back = stack.back();
			
			auto type = expectedType();
			
			using ST = capnp::schema::Type;
			switch(type.which()) {
				case ST::ENUM: {
					auto asEnum = type.asEnum();
					
					// Enums can be decoded from text or integer values
					if(input.getType() == DynamicValue::TEXT) {
						KJ_IF_MAYBE(pEnumerant, asEnum.findEnumerantByName(input.getAs<capnp::Text>())) {
							back.accept(DynamicEnum(*pEnumerant));
							break;
						}
					}
					back.accept(DynamicEnum(asEnum, input.as<uint16_t>>()));
					break;
				}
				
				case ST::DATA:
				case ST::TEXT:
				case ST::LIST:
				case ST::ANY_POINTER:
					back.accept(input);
					break;
				
				case ST::STRUCT: {
					if(input.getType() == DynamicValue::TEXT) {
						auto dst = back.newObject();
						auto schema = dst.getSchema();
						
						auto field = schema.getFieldByName(input.as<capnp::Text>());
						dst.clear(field);
					} else {
						back.accept(input);
					}
					break;
				}
				
				case ST::VOID:
					back.accept(capnp::Void());
					break;
				
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
									
						back.accept(std::stod(asString));
					} else {
						back.accept(input);
					}
					break;
				
				case ST::UINT8:
				case ST::UINT16:
				case ST::UINT32:
				case ST::UINT64: {
					if(input.getType() == DynamicValue::TEXT) {
						back.accept(input.as<capnp::Text>().parseAs<uint64_t>());
					} else {
						back.accept(input);
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
								back.accept(result); \
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
						back.accept(input);
					}
					break;
			}	
		}
	
	private:
		void beginNull() {
			stack.back().accept(nullptr);
			++ignoreDepth;
		}
	};
	
	// Forward declarations
	void loadStruct(DynamicStruct::Builder dst, Cursor& cursor);
	void loadList(capnp::ListSchema listSchema, kj::Function<capnp::DynamicList::Builder(size_t)> initializer, Cursor& cursor);
	
	void loadList(capnp::ListSchema listSchema, kj::Function<capnp::DynamicList::Builder(size_t)> initializer, Cursor& cursor) {
		auto elType = listSchema.getElementType();
		
		Event& arrayOpen = cursor.current();
		KJ_REQUIRE(arrayOpen.event_type() == EventType::begin_array);
		size_t initialSize = arrayOpen.size();
		cursor.next();
		
		// Unfortunately, we do not know the size of the list beforehand in JSON, but we need
		// to know the list size to allocate the output buffer. We allocate a temporary list
		// that we extend as needed
		capnp::MallocMessageBuilder tmpMessage;
		capnp::Orphanage tmpOrphanage = tmpMessage.getOrphanage();
		
		const size_t baseSize = elType.isBool() ? 64 : 8;
		if(initialSize == 0)
			initialSize = baseSize;
		
		using O = capnp::Orphan<capnp::DynamicList>;
		
		kj::Vector<O> previous;
		
		capnp::Orphan<capnp::DynamicList> storage = tmpOrphanage.newOrphan(listSchema, initialSize);
		capnp::DynamicList::Builder dst = storage.get();
		
		size_t storageUsed = 0;
		
		// Process all entries
		while(true) {			
			Event& current = cursor.current();
			if(current.event_type() == EventType::end_array) {
				cursor.next();
				break;
			}
				
			// Make sure we can fit the index
			if(storageUsed >= dst.size()) {
				previous.add(mv(storage));
				
				storage = tmpOrphanage.newOrphan(listSchema, 2 * dst.size());
				dst = storage.get();
				
				storageUsed = 0;
			}
			
			using ST = capnp::schema::Type;
			switch(elType.which()) {
				case ST::STRUCT:
					loadStruct(dst[storageUsed].as<DynamicStruct>(), cursor);
					break;
					
				case ST::LIST: {
					auto initializer = [&](size_t size) {
						return dst.init(storageUsed, size).as<DynamicList>();
					};
					
					loadList(elType.asList(), initializer, cursor);
					break;
				}
				
				case ST::TEXT: {
					auto strView = current.get<jsoncons::string_view>();
					dst.set(storageUsed, capnp::Text::Reader(strView.data(), strView.size()));
					
					cursor.next();
					break;
				}
				
				case ST::DATA: {
					auto byteView = current.get<jsoncons::byte_string_view>();
					dst.set(storageUsed, capnp::Data::Reader(byteView.data(), byteView.size()));
					cursor.next();
					break;
				}
				
				default: {
					dst.set(storageUsed, parsePrimitive(elType, current));
					cursor.next();
					break;
				}
			}
			
			++storageUsed;
		}
		
		// Initialize target orphan
		size_t totalSize = 0;
		for(auto& p : previous)
			totalSize += p.get().size();
		totalSize += storageUsed;
		
		// Copy all data over
		DynamicList::Builder builder = initializer(totalSize);
		
		size_t outIdx = 0;
		for(auto& p : previous) {
			auto reader = p.getReader();
			for(auto el : reader)
				builder.set(outIdx++, el);
		}
		
		for(auto i : kj::range(0, storageUsed)) {
			builder.set(outIdx++, dst[i].asReader());
		}
	}
	
	void loadStruct(DynamicStruct::Builder dst, Cursor& cursor) {
		Event& initial = cursor.current();
		
		// If the node is scalar, this means we set that field with default value
		if(initial.event_type() != EventType::begin_object) {
			auto strView = initial.get<jsoncons::string_view>();
			kj::StringPtr strPtr(strView.data(), strView.size());
			
			auto field = dst.getSchema().getFieldByName(strPtr);
			dst.clear(field);
			
			cursor.next();
			
			return;
		}
		
		cursor.next();
		
		Maybe<kj::String> previouslySetUnionField = nullptr;
		
		while(true) {
			Event& keyEvt = cursor.current();
			if(keyEvt.event_type() == EventType::end_object) {
				cursor.next();
				break;
			}
			
			KJ_REQUIRE(keyEvt.event_type() == EventType::key, "Expected key");
			std::string key = keyEvt.get<std::string>();
			cursor.next();
			
			Event& valueEvt = cursor.current();
			
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
								
				using ST = capnp::schema::Type;
				switch(type.which())  {
					case ST::STRUCT:
						loadStruct(dst.init(field).as<DynamicStruct>(), cursor);
						break;
						
					case ST::LIST: {
						auto initializer = [&](size_t size) {
							return dst.init(field, size).as<DynamicList>();
						};
						loadList(type.asList(), initializer, cursor);
						
						break;
					}
					
					case ST::TEXT: {
						auto strView = valueEvt.get<jsoncons::string_view>();
						dst.set(field, capnp::Text::Reader(strView.data(), strView.size()));
						
						cursor.next();
						break;
					}
					
					case ST::DATA: {
						auto byteView = valueEvt.get<jsoncons::byte_string_view>();
						dst.set(field, capnp::Data::Reader(byteView.data(), byteView.size()));
						cursor.next();
						break;
					}
					
					default: {
						dst.set(field, parsePrimitive(type, valueEvt));
						cursor.next();
						break;
					}
				}
			} else {
				KJ_LOG(WARNING, "Could not find matching field for map entry", key);
			}
		}
	}
	
	void writeValue(DynamicValue::Reader src, Encoder& encoder);
	void writeList(DynamicList::Reader src, Encoder& encoder);
	void writeStruct(DynamicStruct::Reader src, Encoder& encoder);
	
	void writeValue(DynamicValue::Reader src, Encoder& encoder) {
		auto type = src.getType();
		
		if(type == DynamicValue::STRUCT) {
			writeStruct(src.as<DynamicStruct>(), encoder);
		} else if(type == DynamicValue::LIST) {
			writeList(src.as<DynamicList>(), encoder);
		} else {
			emitPrimitive(src, encoder);
		}
	}
	
	void writeList(DynamicList::Reader src, Encoder& encoder) {
		encoder.begin_array(src.size());
		
		for(DynamicValue::Reader el : src)
			writeValue(el, encoder);
		
		encoder.end_array();
	}
		
	void writeStruct(DynamicStruct::Reader src, Encoder& encoder) {	
		auto structSchema = src.getSchema();
		
		// If we have no non-union fields and the active union field is a
		// default-valued one, then we write only the field name (reads nicer)
		if(structSchema.getNonUnionFields().size() == 0) {
			auto maybeActive = src.which();
			KJ_IF_MAYBE(pActive, maybeActive) {
				auto& active = *pActive;
				if(!src.has(active, capnp::HasMode::NON_DEFAULT)) {
					encoder.string_value(active.getProto().getName().cStr());
					return;
				}
			}
		}
		
		encoder.begin_object();
		
		auto emitField = [&](capnp::StructSchema::Field field) {
			auto type = field.getType();
			
			encoder.key(field.getProto().getName().cStr());
			writeValue(src.get(field), encoder);
		};
			
		for(auto field : structSchema.getNonUnionFields()) {
			// There is no point in emitting a non-union void field.
			if(field.getType().isVoid())
				continue;
			
			// Don't emit interface- or any-typed fields
			if(field.getType().isInterface() || field.getType().isAnyPointer())
				continue;
			
			emitField(field);
		}
		
		KJ_IF_MAYBE(pField, src.which()) {
			emitField(*pField);
		}
		
		encoder.end_object();
	}
	
	Own<Cursor> makeCursor(kj::BufferedInputStream& stream, const JsonOptions& options) {
		switch(options.dialect) {
			case JsonOptions::JSON: {
				using JsonCursor = jsoncons::basic_json_cursor<char, JsonSource>;
				
				using DecodeOptions = jsoncons::basic_json_options<char>;
				DecodeOptions opts;
				
				KJ_REQUIRE(options.quoteSpecialNums, "Can not parse unquoted JSON special numbers");
				
				if(options.jsonNan != nullptr)
					opts.nan_to_str(options.jsonNan);
				
				if(options.jsonInf != nullptr)
					opts.inf_to_str(options.jsonInf);
				
				if(options.jsonNegInf != nullptr)
					opts.neginf_to_str(options.jsonNegInf);
				
				auto src = kj::heap<JsonSource>(stream);
				auto cur = kj::heap<JsonCursor>(*src, opts);
				
				return cur.attach(mv(src));
			}
			case JsonOptions::CBOR: {
				using CborCursor = jsoncons::cbor::basic_cbor_cursor<CborSource>;
				
				auto src = kj::heap<CborSource>(stream);
				auto cur = kj::heap<CborCursor>(*src);
				
				return cur.attach(mv(src));
			}
			case JsonOptions::BSON: {
				using BsonCursor = jsoncons::bson::basic_bson_cursor<CborSource>;
				
				auto src = kj::heap<CborSource>(stream);
				auto cur = kj::heap<BsonCursor>(*src);
				
				return cur.attach(mv(src));
			}
		}
		KJ_FAIL_REQUIRE("Unknown dialect");
	}
	
	Own<Encoder> makeEncoder(kj::BufferedOutputStream& stream, const JsonOptions& options) {
		switch(options.dialect) {
			case JsonOptions::JSON: {
				using JsonEncoder = jsoncons::basic_json_encoder<char, JsonSink>;
				
				using EncodeOptions = jsoncons::basic_json_options<char>;
				EncodeOptions opts;
				
				if(options.quoteSpecialNums) {
					opts
						.nan_to_str(options.jsonNan)
						.inf_to_str(options.jsonInf)
						.neginf_to_str(options.jsonNegInf)
					;
				} else {
					opts
						.nan_to_num(options.jsonNan)
						.inf_to_num(options.jsonInf)
						.neginf_to_num(options.jsonNegInf)
					;
				}
				
				
				auto encoder = kj::heap<JsonEncoder>(JsonSink(stream), opts);
				
				return encoder;
			}
			case JsonOptions::CBOR: {
				using CborEncoder = jsoncons::cbor::basic_cbor_encoder<CborSink>;
				
				auto encoder = kj::heap<CborEncoder>(CborSink(stream));
				
				return encoder;
			}
			case JsonOptions::BSON: {
				using BsonEncoder = jsoncons::bson::basic_bson_encoder<CborSink>;
				
				auto encoder = kj::heap<BsonEncoder>(CborSink(stream));
				
				return encoder;
			}
		}
		KJ_FAIL_REQUIRE("Unknown dialect");
	}
}

void loadJson(capnp::DynamicStruct::Builder dst, kj::BufferedInputStream& stream, const JsonOptions& options) {	
	loadStruct(dst, *makeCursor(stream, options));
}

void loadJson(capnp::ListSchema schema, kj::Function<capnp::DynamicList::Builder(size_t)> slot, kj::BufferedInputStream& stream, const JsonOptions& options) {	
	loadList(schema, mv(slot), *makeCursor(stream, options));
}

capnp::DynamicValue::Reader loadJsonPrimitive(capnp::Type type, kj::BufferedInputStream& stream, const JsonOptions& options) {
	return parsePrimitive(type, makeCursor(stream, options) -> current());
}

void writeJson(capnp::DynamicValue::Reader src, kj::BufferedOutputStream& stream, const JsonOptions& options) {		
	auto encoder = makeEncoder(stream, options);
	
	writeValue(src, *encoder);
	encoder -> flush();
}


}