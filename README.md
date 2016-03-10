# GoldFish
## A fast JSON and CBOR streaming library, without using memory

## Why GoldFish?
GoldFish can parse and generate very large [JSON](http://json.org) or [CBOR](http://cbor.io) documents.
It has some similarities to a [SAX](https://en.wikipedia.org/wiki/Simple_API_for_XML) parser, but doesn't use an event driven API, instead the user of the GoldFish interface is in control.
GoldFish intends to be the easiest and fastest JSON and CBOR streaming parser and serializer to use (even though not necessarily the fastest/easiest DOM parser/generator).

## Quick tutorial
### Converting a JSON stream to a CBOR stream
~~~~~~~~~~cpp
#include <goldfish/stream.h>
#include <goldfish/json_reader.h>
#include <goldfish/cbor_writer.h>

int main()
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
~~~~~~~~~~

### Parsing a JSON document with a schema
SAX parsers are notoriably more complicated to use than DOM parser. The order of the fields in a JSON object matters for a SAX parser.
Defining a schema (which is simply an ordering of the expected key names in the object) helps keep the code simple.
Note that the example below is O(1) in memory (meaning the amount of memory used does not depend on the size of the document)

~~~~~~~~~~cpp
#include <goldfish/stream.h>
#include <goldfish/json_reader.h>
#include <goldfish/schema.h>

int main()
{
	using namespace goldfish;

	static const schema s{ "a", "b", "c" };
	auto document = filter_map(json::read(stream::read_string_literal("{\"a\":1,\"b\":3.0}")).as<tags::map>(), s);

	assert(document.read_value("a")->as<tags::unsigned_int>() == 1);
	assert(document.read_value("b")->as<tags::floating_point>() == 3.0);
	assert(document.read_value("c") == nullopt);
	skip(document);
}
~~~~~~~~~~

## Documentation
### Streams
Goldfish parses documents from read streams and serializes documents to write streams.

Goldfish comes with a few readers: a reader over an in memory buffer (see stream::read_buffer_ref) or over a file (see stream::file_reader). It also provides a buffering (see stream::buffer). You might find yourself in a position where you want to implement your own stream, for example, as a network stream on top of your favorite network library.
Not to worry, the interface for a read stream is fairly straightforward, with a single read_buffer API:
~~~~~~~~~~cpp
struct read_stream
{
	// Copies some bytes from the stream to the "buffer"
	// Returns the number of bytes copied.
	// If the API returns something else than buffer.size(), the end of stream was reached.
	// Can throw on IO error.
	//
	// buffer_ref is an object that contains a pointer to the buffer (buffer.data() is the pointer)
	// as well as the number of bytes in the buffer (buffer.size())
	size_t read_buffer(buffer_ref buffer);
}
~~~~~~~~~~

