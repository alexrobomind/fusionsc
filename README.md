# GoldFish
## A fast JSON and CBOR streaming library, without using memory

## Why GoldFish?
GoldFish can parse and generate very large [JSON](http://json.org) or [CBOR](http://cbor.io) documents.
It has some similarities to a [SAX](https://en.wikipedia.org/wiki/Simple_API_for_XML) parser, but doesn't use an event driven API, instead the user of the GoldFish interface is in control.
GoldFish intends to be the easiest and one of the fastest JSON and CBOR streaming parser and serializer to use.

## Quick tutorial
### Converting a JSON stream to a CBOR stream
```cpp
#include <goldfish/json_reader.h>
#include <goldfish/cbor_writer.h>

int main()
{
	using namespace goldfish;

	// Read the string literal as a stream and parse it as a JSON document
	// This doesn't really do any work, the stream will be read as we parse the document
	auto document = json::read(stream::read_string("{\"A\":[1,2,3],\"B\":true}"));

	// Generate a stream on a vector, a CBOR writer around that stream and write
	// the JSON document to it
	// Note that all the streams need to be flushed to ensure that any potentially
	// buffered data is serialized.
	auto cbor_document = cbor::create_writer(stream::vector_writer{}).write(document);
	assert(cbor_document == std::vector<byte>{
		0xbf,                    // start map
		0x61,0x41,               // key: "A"
		0x9f,0x01,0x02,0x03,0xff,// value : [1, 2, 3]
		0x61,0x42,               // key : "B"
		0xf5,                    // value : true
		0xff                     // end map
	});
}
```

### Parsing a JSON document with a schema
SAX parsers are notoriously more complicated to use than DOM parser. The order of the fields in a JSON object matters for a SAX parser.
Defining a schema (which is simply an ordering of the expected key names in the object) helps keep the code simple.
Note that the example below is O(1) in memory (meaning the amount of memory used does not depend on the size of the document)

```cpp
#include <goldfish/json_reader.h>

int main()
{
	using namespace goldfish;

	auto document = json::read(stream::read_string("{\"a\":1,\"c\":3.5}")).as_map("a", "b", "c");
	assert(document.read("a")->as_uint64() == 1);
	assert(document.read("b") == nullopt);
	assert(document.read("c")->as_double() == 3.5);
	seek_to_end(document);
}
```

How about a more complicated example. Note again that this program doesn't allocate memory to parse the document and could run on very large documents backed by file (using `stream::file_reader`) or other type of stream, even on resource constrained machines.

```cpp
#include <goldfish/json_reader.h>
#include <goldfish/iostream_adaptor.h> // to be able to output streams to cout
#include <iostream>

int main()
{
	using namespace goldfish;

	auto document = json::read(stream::read_string(
		R"([
			{"name":"Alice","friends":["Bob","Charlie"]},
			{"name":"Bob","friends":["Alice"]}
		])")).as_array();

	while (auto entry_document = document.read())
	{
		auto entry = entry_document->as_map("name", "friends");
		std::cout << entry.read("name").value().as_string() << " has the following friends: ";

		auto friends = entry.read("friends").value().as_array();
		while (auto friend_name = friends.read())
			std::cout << friend_name->as_string() << " ";

		std::cout << "\n";
		seek_to_end(entry);
	}
	
	/*
	This program outputs:
		Alice has the following friends: Bob Charlie
		Bob has the following friends: Alice
	*/
}
```

### Generating a JSON or CBOR document
You can get a JSON or CBOR writer by calling `json::create_writer` or `cbor::create_writer` on an output stream.

```cpp
#include <goldfish/json_writer.h>

int main()
{
	using namespace goldfish;
	
	auto map = json::create_writer(stream::string_writer{}).start_map();
	map.write("A", 1);
	map.write("B", "text");
	// Streams are serialized as binary 64 data in JSON
	map.write("C", stream::read_string("Hello world!"));

	assert(map.flush() == "{\"A\":1,\"B\":\"text\",\"C\":\"SGVsbG8gd29ybGQh\"}");
}
```

Note how similar the code is to generate a CBOR document. The only change is the creation of the writer (`cbor::create_writer` instead of `json::create_writer`) and the type of output_stream (vector<byte> is better suited to storing the binary data than std::string).
CBOR leads to some significant reduction in document size, in particular when binary data is involved. The JSON document is 41 bytes but the CBOR one is only 27.

```cpp
#include <goldfish/cbor_writer.h>

int main()
{
	using namespace goldfish;

	auto map = cbor::create_writer(stream::vector_writer{}).start_map();
	map.write("A", 1);
	map.write("B", "text");
	map.write("C", stream::read_string("Hello world!"));

	assert(map.flush() == std::vector<byte>{
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
```

## Comparison with other libraries
### Parsing performance
We measured the performance of a trivial task: compute the sum of all the integers in a large JSON document. The rapidjson implementation uses the SAX model of that library. For Casablanca, we had no choice but to load the document as a DOM.
This test was compiled using Visual C++ 2015, ran on an Intel Core i7 CPU, both in 32 and 64 bits, on a 16MB JSON document.
This chart shows the time it took to complete the task, normalized in MB of JSON per second (16MB/duration)

![Parsing comparison](ParsingComparison.png)

Goldfish achieves similar performance to rapidjson (slower on x86 but faster on x64). Both Goldfish and rapidjson are significantly faster than Casablanca, simply because Casablance only offers a DOM interface and couldn't do the job in streaming mode.

### Serialization performance
We loaded the JSON document in a data structure in memory and used the various libraries to regenerate the document in a file on disk.
Both rapidjson and Goldfish used a file stream with a 64kB buffer.

![Serialization comparison](SerializeComparison.png)

Again, Goldfish and rapidjson achieve similar performance (this time Goldfish is faster on x86 but slower on x64).
Those two libraries are again faster than Casablanca mostly because Casablanca doesn't offer a way to generate a JSON document without first creating a DOM in memory.

## Documentation
### Streams
Goldfish parses documents from read streams and serializes documents to write streams.

Goldfish comes with a few readers: a reader over an in memory buffer (see `stream::read_buffer_ref`) or over a file (see `stream::file_reader`). It also provides a buffering (see `stream::buffer`). You might find yourself in a position where you want to implement your own stream, for example, as a network stream on top of your favorite network library.
Not to worry, the interface for a read stream is fairly straightforward, with a single read_partial_buffer API:
```cpp
struct read_stream
{
	// Copies some bytes from the stream to the "buffer"
	// Returns the number of bytes copied, which might be less than buffer.size() if not all the data is immediately available
	// Returns 0 if the buffer is empty or if the stream was at the end before the call was made.
	//
	// buffer_ref is an object that contains a pointer to the buffer (buffer.data() is the pointer)
	// as well as the number of bytes in the buffer (buffer.size())
	size_t read_partial_buffer(buffer_ref buffer);
}
```

Write streams have the following interface:
```cpp
struct write_stream
{
	// Write some data to the stream
	void write_buffer(const_buffer_ref data);

	// Finish writing to the stream
	// This API must be called once the end of stream is reached.
	// It may return some data. For example, a vector_writer returns
	// the data written to the stream (in the form of an std::vector<byte>).
	auto flush();
}
```

There are a few helper APIs that you can use to ease the consumption of streams:
```cpp
// Seek forward in the stream up to cb bytes
// This API returns the number of bytes skipped from the stream, which can be less
// than cb if the end of the stream is reached
// It is implemented in terms of read_partial_buffer, unless the reader_stream has a seek
//  method on it (in which case that method is used)
uint64_t stream::seek(reader_stream&, uint64_t cb);

// Read the entire stream in memory
std::vector<byte> stream::read_all(reader_stream&);
std::string stream::read_all_as_string(reader_stream&);

// Read an object of type T from the stream
// The object must be a POD
// This API is implemented in terms of read_partial_buffer, unless the reader_stream has a
// read method on it (in which case that method is used)
// If the end of stream is reached before sizeof(T) bytes could be read, this method
// throws unexpected_end_of_stream
template <class T> T stream::read(reader_stream&);

// Write an object of type T to the stream
// The object must be a POD
// This API is implemented in terms of write_buffer, unless the writer_stream has a
// write method on it (in which case that method is used)
template <class T> void stream::write(writer_stream&, const T&);

// Copy a reader stream to an output stream
// Note that this API doesn't flush the output stream and returns the writer stream as a convenience
template <class Reader, class Writer> Writer copy(Reader&&, Writer&&);
```

Here is the exhaustive list of readers provided by the library:
* `stream::ref_reader<reader_stream>` (created using `stream::ref(reader_stream&)`): copyable stream that stores a non owning reference to an existing stream
* `stream::const_buffer_ref_reader` (created using `stream::read_buffer_ref`, `stream::read_string_ref` or `stream::read_string` with a string literal): a stream that reads a buffer, without owning that buffer
* `stream::vector_reader` (created using `stream::read_buffer`): a stream that reads an `std::vector<byte>`, owning that vector
* `stream::string_reader` (created using `stream::read_string`): a stream that reads an `std::string`, owning that string
* `stream::base64_reader<reader_stream>` (created using `stream::decode_base64(reader_stream)`): convert a base64 stream into a binary stream
* `stream::buffered_reader<N, reader_stream>` (created using `stream::buffer<N>(reader_stream)`): add an N byte buffer to the reader_stream
* `stream::file_reader`: a reader stream on a file
* `stream::reader_on_reader_writer` (created using `create_reader_writer_stream`): the reader end of a reader/writer (or producer/consumer) stream

Note that those streams can be composed. For example, `stream::decode_base64(stream::buffer<8192>(stream::file_reader("foo.txt")))` opens the file "foo.txt", buffers that stream using an 8kB buffer and decodes the content of the file assuming it is base64 encoded.

Here is the list of writers provided by the library:
* `stream::ref_writer<writer_stream>` (created using `stream::ref(writer_stream&)`): copyable stream that stores a non owning reference to an existing stream
* `stream::vector_writer`: stores the data in memory, in an std::vector<byte>
* `stream::string_writer`: stores the data in memory, in an std::string
* `stream::base64_writer<writer_stream>` (created using `stream::encode_base64_to(writer_stream)`): data written to that stream is base64 encoded before being written to the writer_stream
* `stream::buffered_writer<N, writer_stream>` (created using `stream::buffer<N>(writer_stream)`): add an N byte buffer to the writer_stream
* `stream::file_writer`: a writer stream on a file
* `stream::writer_on_reader_writer` (created using `create_reader_writer_stream`): the writer end of a reader/writer (or producer/consumer) stream

### JSON/CBOR parser
To start the parsing of a read stream use json::read or cbor::read (for JSON or CBOR documents respectively). Those APIs return "document reader" objects.
A document reader offers the following APIs:
* `as_string()`: if the document is a text (for example `"Hello"` in JSON, or an object of major type 3 in CBOR), return a reader stream on the text, otherwise throw `goldfish::bad_variant_access`
* `as_binary()`:
	* For CBOR documents, return a stream on the data of a byte string document (major type 2), or throw `goldfish::bad_variant_access` if the document is not of major type 2.
	* For JSON documents, return a stream that decodes the base64 encoded text if the document is text (for example, if the document is `"SGVsbG8="`, this API returns a stream that reads `Hello`)
* `as_array()`: if the document is an array (for example `[1,"Hello"]` in JSON, or an object of major type 4 in CBOR), return an `array reader` object, otherwise throw `goldfish::bad_variant_access`
* `as_map()`, `as_object()`: if the document is an object (for example `{"Hello":1}` in JSON, or an object of major type 5 in CBOR), return a `map reader` object, otherwise throw `goldfish::bad_variant_access`
* `as_map(...)`, `as_object(...)`: if parameters are specified to `as_map` or `as_object`, a `map reader with schema` object is returned. This allows for simpler parsing of documents when the keys and their order is known in advance.
* `as_double`:
	* if the document is an integer or a floating point (for example `1`, `-1` or `1.0` in JSON), return a double that represents the value of the document.
	* Strings are parsed, which means the JSON document `"8000"` can be read as either the text `8000` using as_text, the text `ÛM4` using as_binary, the double `8000`, the signed integer `8000` or the unsigned integer `8000`
	* otherwise, `goldfish::bad_variant_access` is thrown
* `as_uint64`, `as_uint32`, `as_uint16`, `as_uint8`:
	* if the document is a positive integer (for example `1` in JSON), return an integer that represents the value of the document
	* if the document is a negative integer (for example `-1` in JSON), or if the if the integer is too large to be represented as the requested type, throws `goldfish::integer_overflow_while_casting`
	* Strings are parsed
	* otherwise, `goldfish::bad_variant_access` is thrown
* `as_int64`, `as_int32`, `as_int16`, `as_int8`:
	* if the document is an integer (for example `1` in JSON), return an integer that represents the value of the document, or throws `goldfish::integer_overflow_while_casting` if the value is not representable in the requested type
	* Strings are parsed
	* otherwise, `goldfish::bad_variant_access` is thrown
* `as_bool`: if the document is `true` or `false`, `"true"` or `"false"` return the corresponding boolean value
* `is_null`: return true if the document is `null` in JSON or the equivalent in CBOR (major type 7 and additional information 22).
* `is_undefined_or_null`: return true if the document is null or, for CBOR, undefined

In addition, the document reader implements the visitor pattern and exposes a visit API.
That API calls the provided callback with the object and a tag that represents the semantic type of the object.
Here is an example on how to use that API:

```cpp
#include <iostream>
#include <goldfish/json_reader.h>

using namespace goldfish;

struct my_handler
{
	template <class Stream> const char* operator()(Stream& s, tags::binary) { return "binary"; }
	template <class Stream> const char* operator()(Stream& s, tags::string) { return "string"; }
	template <class ArrayReader> const char* operator()(ArrayReader& s, tags::array) { return "array"; }
	template <class MapReader> const char* operator()(MapReader& s, tags::map) { return "map"; }
	const char* operator()(undefined, tags::undefined) { return "undefined"; }
	const char* operator()(double, tags::floating_point) { return "floating point"; }
	const char* operator()(uint64_t, tags::unsigned_int) { return "uint"; }
	const char* operator()(int64_t, tags::signed_int) { return "int"; }
	const char* operator()(bool, tags::boolean) { return "bool"; }
	const char* operator()(nullptr_t, tags::null) { return "null"; }
};
int main()
{
	my_handler sink;
	std::cout << json::read(stream::read_string("true")).visit(sink);
	// outputs bool, the result of calling sink(true, tags::boolean{})
}
```

For simplicity, you can use `goldfish::best_match` and work with lambdas. `best_match` is an API that takes any number of lambdas and forwards any call to the lambda that has the best matching signature (using the C++ overload resolution rules).

```cpp
#include <iostream>
#include <goldfish/json_reader.h>

int main()
{
	using namespace goldfish;

	std::cout << json::read(stream::read_string("true")).visit(best_match(
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
	));
	// outputs "bool"
}
```

Finally, you could also use `first_match`, which will forward to the first callable lambda. This allows specifying only some of the options:

```cpp
#include <iostream>
#include <goldfish/json_reader.h>

int main()
{
	using namespace goldfish;

	std::cout << json::read(stream::read_string("true")).visit(best_match(
		[](bool, tags::boolean) { return "bool"; },
		[](auto&&, auto) { return "not bool"; }
	));
	// outputs "bool"
}
```