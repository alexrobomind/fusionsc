#include "textio.h"

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

// Shim between jsoncons and our text IO

namespace fsc { namespace textio {

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
	
	Own<Cursor> makeCursor(kj::BufferedInputStream& stream, const Dialect& options) {
		switch(options.language) {
			case Dialect::JSON: {
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
			case Dialect::CBOR: {
				using CborCursor = jsoncons::cbor::basic_cbor_cursor<CborSource>;
				
				jsoncons::cbor::cbor_options opts;
				opts.pack_strings(true);
				
				auto src = kj::heap<CborSource>(stream);
				auto cur = kj::heap<CborCursor>(*src, opts);
				
				return cur.attach(mv(src));
			}
			case Dialect::BSON: {
				using BsonCursor = jsoncons::bson::basic_bson_cursor<CborSource>;
				
				auto src = kj::heap<CborSource>(stream);
				auto cur = kj::heap<BsonCursor>(*src);
				
				return cur.attach(mv(src));
			}
			default:
				KJ_FAIL_REQUIRE("Invalid language");
		}
	}
	
	Own<Encoder> makeEncoder(kj::BufferedOutputStream& stream, const Dialect& options) {
		switch(options.language) {
			case Dialect::JSON: {
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
			case Dialect::CBOR: {
				using CborEncoder = jsoncons::cbor::basic_cbor_encoder<CborSink>;
				
				auto encoder = kj::heap<CborEncoder>(CborSink(stream));
				
				return encoder;
			}
			case Dialect::BSON: {
				using BsonEncoder = jsoncons::bson::basic_bson_encoder<CborSink>;
				
				auto encoder = kj::heap<BsonEncoder>(CborSink(stream));
				
				return encoder;
			}
			default:
				KJ_FAIL_REQUIRE("Invalid language");
		}
	}
	
	struct JsonVisitor : public Visitor {
		Encoder& enc;
		bool allowGenericKeys = false;
		bool isDone = false;
		
		enum State {
			VALUE, MAP_KEY, MAP_VALUE
		};
		kj::Vector<State> states;
		
		JsonVisitor(Encoder& e, bool genericKeys) :
			enc(e),
			allowGenericKeys(genericKeys)
		{
			states.add(VALUE);
			this -> supportsIntegerKeys = genericKeys;
		}
		
		State& state() { KJ_REQUIRE(!states.empty()); return states.back(); }
		
		void checkDone() {
			if(states.size() == 1) {
				isDone = true;
				enc.flush();
			}
		}
		
		void advanceMap(bool strKey) {
			if(state() == MAP_KEY) {
				state() = MAP_VALUE;
				KJ_REQUIRE(allowGenericKeys || strKey, "This language only supports string keys");
			} else if(state() == MAP_VALUE) {
				state() = MAP_KEY;
			}
		}
		
		void beginObject(Maybe<size_t> s) override {
			advanceMap(false);
			states.add(MAP_KEY);
			
			KJ_IF_MAYBE(pSize, s) {
				enc.begin_object(*pSize);
			} else {
				enc.begin_object();
			}
		}
		
		void endObject() override {
			KJ_REQUIRE(state() != MAP_VALUE, "Object ended without receiving value");
			KJ_REQUIRE(state() != VALUE, "Object ended without corresponding beginObject()");
			states.removeLast();
			
			enc.end_object();
			checkDone();
		}
		
		void beginArray(Maybe<size_t> s) override {
			advanceMap(false);
			states.add(VALUE);
			
			KJ_IF_MAYBE(pSize, s) {
				enc.begin_array(*pSize);
			} else {
				enc.begin_array();
			}
		}
		
		void endArray() override {
			KJ_REQUIRE(state() == VALUE, "Array ended without corresponding beginArray()");
			states.removeLast();
			
			enc.end_array();
			checkDone();
		}
		
		void acceptNull() override {
			advanceMap(false);
			enc.null_value();
			checkDone();
		}
		
		void acceptBool(bool v) override {
			advanceMap(false);
			enc.bool_value(v);
			checkDone();
		}
		
		void acceptInt(int64_t v) override {
			advanceMap(false);
			enc.int64_value(v);
			checkDone();
		}
		
		void acceptUInt(uint64_t v) override {
			advanceMap(false);
			enc.uint64_value(v);
			checkDone();
		}
		
		void acceptDouble(double d) override {
			advanceMap(false);
			enc.double_value(d);
			checkDone();
		}
		
		void acceptData(kj::ArrayPtr<const byte> data) override {
			advanceMap(false);
			enc.byte_string_value(jsoncons::byte_string_view(data.begin(), data.size()));
			checkDone();
		}
		
		void acceptString(kj::StringPtr str) override {
			switch(state()) {
				case MAP_VALUE:
				case VALUE:
					enc.string_value(str.cStr());
					break;
				
				case MAP_KEY:
					enc.key(str.cStr());
					break;
			}
			
			advanceMap(true);
			checkDone();
		}
		
		bool done() override {
			return isDone;
		}
	};
	
	void loadInto(Cursor& c, Visitor& v) {
		while(true) {
			if(c.done())
				break;
			
			Event& evt = c.current();
			
			switch(evt.event_type()) {
				case EventType::begin_array: {
					size_t s = evt.size();
					
					if(s != 0)
						v.beginArray(s);
					else
						v.beginArray(nullptr);
					
					break;
				}
				
				case EventType::end_array:
					v.endArray();
					break;
					
				case EventType::begin_object: {
					size_t s = evt.size();
					
					if(s != 0)
						v.beginObject(s);
					else
						v.beginObject(nullptr);
					
					break;
				}
				
				case EventType::end_object:
					v.endObject();
					break;
				
				/*case jsoncons::EventType::key:
					auto str = event.get<std::string>();
					v.acceptKey(str);
					break;*/
				
				case EventType::key:
				case EventType::string_value: {
					auto strView = evt.get<jsoncons::string_view>();
					v.acceptString(kj::heapString(strView.data(), strView.size()));
					break;
				}
				
				case EventType::byte_string_value: {
					auto byteView = evt.get<jsoncons::byte_string_view>();
					v.acceptData(kj::ArrayPtr<const byte>(byteView.data(), byteView.size()));
					break;
				}

				case EventType::null_value:
					v.acceptNull();
					break;
				
				case EventType::bool_value:
					v.acceptBool(evt.get<bool>());
					break;
				
				case EventType::int64_value:
					v.acceptInt(evt.get<int64_t>());
					break;
				
				case EventType::uint64_value:
					v.acceptUInt(evt.get<uint64_t>());
					break;
				
				case EventType::half_value:
				case EventType::double_value:
					v.acceptDouble(evt.get<double>());
					break;
			}
			
			c.next();
		}
	}
}

namespace internal {
	Own<Visitor> createJsonconsWriter(kj::BufferedOutputStream& stream, const Dialect& dialect) {
		auto encoder = makeEncoder(stream, dialect);
		auto writer = kj::heap<JsonVisitor>(*encoder, /*dialect.language == Dialect::CBOR*/false);
		return writer.attach(mv(encoder));
	}

	void jsonconsLoad(kj::BufferedInputStream& stream, Visitor& v, const Dialect& dialect) {
		auto cursor = makeCursor(stream, dialect);
		loadInto(*cursor, v);
	}
}

}}