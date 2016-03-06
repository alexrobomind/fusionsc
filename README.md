# GoldFish
## A fast JSON and CBOR streaming library, without using memory

## Why GoldFish?
GoldFish can parse and generate very large [JSON](http://json.org) or [CBOR](http://cbor.io) documents.
It has some similarities to a [SAX](https://en.wikipedia.org/wiki/Simple_API_for_XML) parser, but doesn't use an event driven API, instead the user of the GoldFish interface is in control.

## Quick tutorial
### Converting a JSON stream to a CBOR stream
~~~~~~~~~~cpp
#include <gold_fish/stream.h>
#include <gold_fish/json_reader.h>
#include <gold_fish/cbor_writer.h>

int main()
{
	using namespace gold_fish;

	// Read the string literal as a stream and parse it as a JSON document
	// This doesn't really do any work, the stream will be read as we parse the document
	// The stream needs to stay alive until we are done copying the data!
	auto input_stream = stream::read_string_literal("{\"a\":[1,2,3],\"b\":3.0}");
	auto document = json::read(input_stream);

	// Generate a stream on a vector, a CBOR writer around that stream and write
	// the JSON document to it
	// Note that all the streams need to be flushed to ensure that there any potentially
	// buffered data is serialized.
	stream::vector_writer output_stream;
	cbor::write(output_stream).write(document);
	output_stream.flush();
}
~~~~~~~~~~
