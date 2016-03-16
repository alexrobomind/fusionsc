#include "unit_test.h"

#include <goldfish/stream.h>
#include <goldfish/json_reader.h>
#include <goldfish/cbor_writer.h>

TEST_CASE(convert_json_to_cbor)
{
	using namespace goldfish;

	// Read the string literal as a stream and parse it as a JSON document
	// This doesn't really do any work, the stream will be read as we parse the document
	auto document = json::read(stream::read_string_literal("{\"a\":[1,2,3],\"b\":3.0}"));

	// Generate a stream on a vector, a CBOR writer around that stream and write
	// the JSON document to it
	// Note that all the streams need to be flushed to ensure that there any potentially
	// buffered data is serialized.
	stream::vector_writer output_stream;
	copy_sax_document(cbor::create_writer(stream::ref(output_stream)), document);
	output_stream.flush();

	// output_stream.data contains the CBOR document
}

#include <goldfish/stream.h>
#include <goldfish/json_reader.h>
#include <goldfish/schema.h>

TEST_CASE(parse_document)
{
	using namespace goldfish;

	static const schema s{ "a", "b", "c" };
	auto document = apply_schema(json::read(stream::read_string_literal("{\"a\":1,\"c\":3.0}")).as_map(), s);

	test(document.read_value("a")->as_uint() == 1);
	test(document.read_value("b") == nullopt);
	test(document.read_value("c")->as_double() == 3.0);
	seek_to_end(document);
}

#include <goldfish/dom_writer.h>
#include <goldfish/json_writer.h>
#include <goldfish/cbor_writer.h>

TEST_CASE(generate_json_document)
{
	using namespace goldfish;
	
	stream::string_writer output_stream;
	auto map = json::create_writer(stream::ref(output_stream)).start_map();
	map.write_key(1);
	map.write_value(2);
	map.flush();
	output_stream.flush();
	test(output_stream.data == "{1:2}");
}
TEST_CASE(generate_cbor_document)
{
	using namespace goldfish;

	stream::vector_writer output_stream;
	cbor::create_writer(stream::ref(output_stream)).write(true);
	output_stream.flush();
}