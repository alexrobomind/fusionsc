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
	cbor::write(stream::ref(output_stream)).write(document);
	output_stream.flush();
}

#include <goldfish/stream.h>
#include <goldfish/json_reader.h>
#include <goldfish/schema.h>

TEST_CASE(parse_document)
{
	using namespace goldfish;

	static const schema s{ "a", "b", "c" };
	auto document = filter_map(json::read(stream::read_string_literal("{\"a\":1,\"b\":3.0}")).as<tags::map>(), s);

	test(document.read_value("a")->as<tags::unsigned_int>() == 1);
	test(document.read_value("b")->as<tags::floating_point>() == 3.0);
	test(document.read_value("c") == nullopt);
	skip(document);
}