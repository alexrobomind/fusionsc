#pragma once

#include "stream.h"
#include <memory>

namespace goldfish { namespace stream
{
	class typed_erased_reader
	{
	public:
		template <class inner> typed_erased_reader(inner&& stream)
			: m_reader(std::make_unique<reader_impl<std::decay_t<inner>>>(std::forward<inner>(stream)))
		{}

		size_t read_buffer(buffer_ref buffer) { return get().read_buffer(buffer); }
		uint64_t seek(uint64_t cb) { return get().seek(cb); }

	private:
		struct reader_interface
		{
			virtual ~reader_interface() = default;
			virtual size_t read_buffer(buffer_ref) = 0;
			virtual uint64_t seek(uint64_t) = 0;
		};
		template <class inner> class reader_impl : public reader_interface
		{
		public:
			reader_impl(inner stream)
				: m_stream(std::move(stream))
			{}
			size_t read_buffer(buffer_ref buffer) override { return m_stream.read_buffer(buffer); }
			uint64_t seek(uint64_t cb) override { return stream::seek(m_stream, cb); }

		private:
			inner m_stream;
		};
		reader_interface& get() { return *m_reader; }
		std::unique_ptr<reader_interface> m_reader;
	};
	template <class inner> enable_if_reader_t<inner, typed_erased_reader> erase_type(inner&& stream) { return{ std::forward<inner>(stream) }; }
}}
