#include "unit_test.h"

#include <goldfish/json_reader.h>
#include <goldfish/cbor_writer.h>

TEST_CASE(convert_json_to_cbor)
{
	using namespace goldfish;

	// Read the string literal as a stream and parse it as a JSON document
	// This doesn't really do any work, the stream will be read as we parse the document
	auto document = json::read(stream::read_string("{\"A\":[1,2,3],\"B\":true}"));

	// Generate a stream on a vector, a CBOR writer around that stream and write
	// the JSON document to it
	// Note that all the streams need to be flushed to ensure that there any potentially
	// buffered data is serialized.
	auto cbor_document = cbor::create_writer(stream::vector_writer{}).write(document);
	test(cbor_document == std::vector<byte>{
		0xbf,                    // start map
		0x61,0x41,               // key: "A"
		0x9f,0x01,0x02,0x03,0xff,// value : [1, 2, 3]
		0x61,0x42,               // key : "B"
		0xf5,                    // value : true
		0xff                     // end map
	});
}

#include <sstream>
#include <goldfish/json_reader.h>

TEST_CASE(parse_simple)
{
	using namespace goldfish;

	auto document = json::read(stream::read_string("{\"a\":1,\"c\":3.5}")).as_map("a", "b", "c");
	assert(document.read("a").value().as_uint64() == 1);
	assert(document.read("b") == nullopt);
	assert(document.read("c").value().as_double() == 3.5);
	seek_to_end(document);
}

#include <goldfish/iostream_adaptor.h> // to be able to output streams to cout

TEST_CASE(parse_complex)
{
	using namespace goldfish;

	auto document = json::read(stream::read_string(
		R"([
			{"name":"Alice","friends":["Bob","Charlie"]},
			{"name":"Bob","friends":["Alice"]}
		])")).as_array();

	std::stringstream output;
	while (auto entry_document = document.read())
	{
		auto entry = entry_document->as_map("name", "friends");
		output << entry.read("name").value().as_string() << " has the following friends: ";

		auto friends = entry.read("friends").value().as_array();
		while (auto friend_name = friends.read())
			output << friend_name->as_string() << " ";

		output << "\n";
		seek_to_end(entry);
	}

	test(output.str() ==
		"Alice has the following friends: Bob Charlie \n"
		"Bob has the following friends: Alice \n");
}

#include <goldfish/json_writer.h>

TEST_CASE(generate_json_document)
{
	using namespace goldfish;
	
	auto map = json::create_writer(stream::string_writer{}).start_map();
	map.write("A", 1);
	map.write("B", "text");
	map.write("C", stream::read_string("Hello world!"));

	// Streams are serialized as binary 64 data in JSON
	test(map.flush() == "{\"A\":1,\"B\":\"text\",\"C\":\"SGVsbG8gd29ybGQh\"}");
}

#include <goldfish/cbor_writer.h>

TEST_CASE(generate_cbor_document)
{
	using namespace goldfish;

	auto map = cbor::create_writer(stream::vector_writer{}).start_map();
	map.write("A", 1);
	map.write("B", "text");
	map.write("C", stream::read_string("Hello world!"));

	test(map.flush() == std::vector<byte>{
		0xbf,                               // start map marker
		0x61,0x41,                          // key: "A"
		0x01,                               // value : uint 1
		0x61,0x42,                          // key : "B"
		0x64,0x74,0x65,0x78,0x74,           // value : "text"
		0x61,0x43,                          // key : "C"
		0x4c,0x48,0x65,0x6c,0x6c,0x6f,0x20,
		0x77,0x6f,0x72,0x6c,0x64,0x21,      // value : binary blob "Hello world!"
		0xff                                // end of map
	});
}

struct my_handler
{
	template <class Stream> const char* operator()(Stream& s, goldfish::tags::binary) { return "binary"; }
	template <class Stream> const char* operator()(Stream& s, goldfish::tags::string) { return "string"; }
	template <class ArrayReader> const char* operator()(ArrayReader& s, goldfish::tags::array) { return "array"; }
	template <class MapReader> const char* operator()(MapReader& s, goldfish::tags::map) { return "map"; }
	const char* operator()(goldfish::undefined, goldfish::tags::undefined) { return "undefined"; }
	const char* operator()(double, goldfish::tags::floating_point) { return "floating point"; }
	const char* operator()(uint64_t, goldfish::tags::unsigned_int) { return "uint"; }
	const char* operator()(int64_t, goldfish::tags::signed_int) { return "int"; }
	const char* operator()(bool, goldfish::tags::boolean) { return "bool"; }
	const char* operator()(nullptr_t, goldfish::tags::null) { return "null"; }
};

TEST_CASE(test_visit)
{
	using namespace goldfish;
	my_handler sink;
	test(json::read(stream::read_string("true")).visit(sink) == "bool");
}

TEST_CASE(test_visit_with_best_match)
{
	using namespace goldfish;
	test(json::read(stream::read_string("true")).visit(best_match(
		[](auto&&, tags::binary) { return "binary"; },
		[](auto&&, tags::string) { return "string"; },
		[](auto&&, tags::array) { return "array"; },
		[](auto&&, tags::map) { return "map"; },
		[](undefined, tags::undefined) { return "undefined"; },
		[](double, tags::floating_point) { return "floating point"; },
		[](uint64_t, tags::unsigned_int) { return "uint"; },
		[](int64_t, tags::signed_int) { return "int"; },
		[](bool, tags::boolean) { return "bool"; },
		[](nullptr_t, tags::null) { return "null"; }
	)) == "bool");
}

TEST_CASE(test_visit_with_first_match)
{
	using namespace goldfish;
	test(json::read(stream::read_string("true")).visit(first_match(
		[](bool, tags::boolean) { return "bool"; },
		[](auto&&, auto) { return "not a bool"; }
	)) == "bool");
}