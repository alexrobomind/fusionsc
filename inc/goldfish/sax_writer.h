#pragma once

#include "match.h"
#include "tags.h"
#include <type_traits>

namespace goldfish {
	template <class DocumentWriter, class Document>
	std::enable_if_t<tags::has_tag<std::decay_t<Document>, tags::document>::value, void> copy_document(DocumentWriter&& writer, Document&& document)
	{
		auto copy_stream = [](auto& s, auto&& output_stream_generator_known_size, auto&& output_stream_generator_unknown_size)
		{
			uint8_t buffer[8 * 1024];
			auto cb = s.read_buffer(buffer);
			if (cb < sizeof(buffer))
			{
				// We read the entire stream
				auto output_stream = output_stream_generator_known_size(cb);
				output_stream.write_buffer({ buffer, cb });
				output_stream.flush();
			}
			else
			{
				// We read only a portion of the stream
				auto output_stream = output_stream_generator_unknown_size();
				output_stream.write_buffer(buffer);
				do
				{
					cb = s.read_buffer(buffer);
					output_stream.write_buffer({ buffer, cb });
				} while (cb == sizeof(buffer));
				output_stream.flush();
			}
		};

		document.visit(first_match(
			[&](auto&& x, tags::binary)
			{
				copy_stream(x, [&](auto cb) { return writer.write_binary(cb); }, [&] { return writer.write_binary(); });
			},
			[&](auto&& x, tags::string)
			{
				copy_stream(x, [&](auto cb) { return writer.write_text(cb); }, [&] { return writer.write_text(); });
			},
			[&](auto&& x, tags::array)
			{
				auto array_writer = writer.write_array();
				while (auto element = x.read())
					copy_document(array_writer.append(), *element);
				array_writer.flush();
			},
			[&](auto&& x, tags::map)
			{
				auto map_writer = writer.write_map();
				while (auto key = x.read_key())
				{
					copy_document(map_writer.append_key(), *key);
					copy_document(map_writer.append_value(), x.read_value());
				}
				map_writer.flush();
			},
			[&](auto&& x, tags::undefined) { writer.write(x); },
			[&](auto&& x, tags::floating_point) { writer.write(x); },
			[&](auto&& x, tags::unsigned_int) { writer.write(x); },
			[&](auto&& x, tags::signed_int) { writer.write(x); },
			[&](auto&& x, tags::boolean) { writer.write(x); },
			[&](auto&& x, tags::null) { writer.write(x); }
		));
	}
}
