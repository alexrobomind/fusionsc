#include <iostream>
#include <numeric>
#include <chrono>

#include <goldfish/stream.h>
#include <goldfish/file_stream.h>
#include <goldfish/json_reader.h>
#include <goldfish/json_writer.h>
#include <goldfish/cbor_reader.h>
#include <goldfish/cbor_writer.h>

using namespace std;
using namespace goldfish;

template <class Lambda>
auto measure_one(const Lambda& l)
{
	auto before = chrono::high_resolution_clock::now();
	l();
	return chrono::high_resolution_clock::now() - before;
}

template <class Lambda>
void measure(Lambda&& l, size_t document_size)
{
	vector<chrono::milliseconds> durations;
	auto start = chrono::high_resolution_clock::now();
	do
	{
		durations.push_back(chrono::duration_cast<chrono::milliseconds>(measure_one(l)));
	} while (chrono::high_resolution_clock::now() - start < chrono::seconds(10));

	sort(durations.begin(), durations.end());
	double average_duration = (double)accumulate(durations.begin(), durations.end(), chrono::milliseconds(0)).count() / durations.size();
	cout << "average: " << average_duration << "ms (" << document_size / (average_duration * 1000) << "MB/s) on " << durations.size() << " samples\n";
	cout << "best: " << durations.front().count() << "ms\tworst: " << durations.back().count() << "ms\n";
}

template <class Document> int64_t sum_ints(Document&& t)
{
	return t.visit(first_match(
		[](auto x, tags::unsigned_int) -> int64_t { return x; },
		[](auto x, tags::signed_int) { return x; },
		[](auto& x, tags::array)
		{
			int64_t sum = 0;
			while (auto d = x.read())
				sum += sum_ints(*d);
			return sum;
		},
		[](auto& x, tags::map)
		{
			int64_t sum = 0;
			while (auto key = x.read_key())
			{
				goldfish::seek_to_end(*key);
				sum += sum_ints(x.read_value());
			}
			return sum;
		},
		[](auto& x, auto tag) { goldfish::seek_to_end(x); return 0ll; }));
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "This program requires a single argument which is the path to a JSON file";
		return 0;
	}

	auto json_data = stream::read_all(stream::file_reader(argv[1]));
	auto cbor_data = [&]
	{
		auto document = json::read(stream::read_buffer_ref(json_data));

		stream::vector_writer output_stream;
		cbor::create_writer(stream::ref(output_stream)).write(document);
		return output_stream.flush();
	}();

	cout << "\nSTREAMING MODE\n";

	cout << "\nDeserialize CBOR in streaming mode\n";
	measure([&]
	{
		return sum_ints(cbor::read(stream::read_buffer_ref(cbor_data)));
	}, cbor_data.size());

	cout << "\nDeserialize JSON in streaming mode\n";
	measure([&]
	{
		return sum_ints(json::read(stream::read_buffer_ref(json_data)));
	}, json_data.size());
}

