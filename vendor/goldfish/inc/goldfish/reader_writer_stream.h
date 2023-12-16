#pragma once

#include "array_ref.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace goldfish { namespace stream
{
	struct reader_writer_stream_closed : exception { using exception::exception; };

	namespace details
	{
		/* This acts in a similar manner to a producer consumer queue */
		class reader_writer_stream
		{
		public:
			size_t read_partial_buffer(buffer_ref data)
			{
				auto original_size = data.size();
				if (original_size == 0)
					return 0;

				std::unique_lock<std::mutex> lock(m_mutex);
				assert(m_state != state::opened || m_read_buffer == nullptr);
				m_read_buffer = &data;
				m_condition_variable.notify_one(); // Wake up the writer (now that m_read_buffer is set)
				m_condition_variable.wait(lock, [&] { return m_state != state::opened || m_read_buffer == nullptr; });
				if (m_state == state::terminated)
					throw reader_writer_stream_closed{ "Failed to read from the reader writer stream because the writer was closed" };
				return original_size - data.size();
			}

			void write_buffer(const_buffer_ref data)
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				assert(m_state != state::flushed);

				while (!data.empty())
				{
					m_condition_variable.wait(lock, [&] { return m_state == state::terminated || m_read_buffer != nullptr; });
					if (m_state == state::terminated)
						throw reader_writer_stream_closed{ "Failed to write to the reader writer stream because the reader was closed" };
					assert(m_read_buffer != nullptr);

					auto to_copy = std::min(m_read_buffer->size(), data.size());
					copy(data.remove_front(to_copy), m_read_buffer->remove_front(to_copy));
					m_read_buffer = nullptr;

					// Wake up the reader now that m_read_buffer is null
					m_condition_variable.notify_one();
				}
			}
			void flush()
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				assert(m_state != state::flushed);
				if (m_state == state::terminated)
					throw reader_writer_stream_closed{ "Failed to flush to the reader writer stream because the reader was closed" };

				m_state = state::flushed;
				m_condition_variable.notify_one();
			}
			void terminate()
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				if (m_state == state::opened)
				{
					m_state = state::terminated;
					m_condition_variable.notify_one();
				}
			}

		private:
			std::mutex m_mutex;
			std::condition_variable m_condition_variable;
			buffer_ref* m_read_buffer;

			enum class state
			{
				opened,
				flushed,
				terminated,
			} m_state = state::opened;
		};
	}

	class reader_on_reader_writer
	{
	public:
		reader_on_reader_writer(std::shared_ptr<details::reader_writer_stream> stream)
			: m_stream(std::move(stream))
		{}
		reader_on_reader_writer(const reader_on_reader_writer&) = delete;
		reader_on_reader_writer(reader_on_reader_writer&& rhs)
			: m_stream(std::move(rhs.m_stream))
		{
			rhs.m_stream = nullptr;
		}
		reader_on_reader_writer& operator = (const reader_on_reader_writer&) = delete;
		reader_on_reader_writer& operator = (reader_on_reader_writer&& rhs)
		{
			std::swap(m_stream, rhs.m_stream);
			return *this;
		}

		~reader_on_reader_writer()
		{
			if (m_stream)
				m_stream->terminate();
		}
		size_t read_partial_buffer(buffer_ref data)
		{
			return m_stream->read_partial_buffer(data);
		}

	private:
		std::shared_ptr<details::reader_writer_stream> m_stream;
	};
	class writer_on_reader_writer
	{
	public:
		writer_on_reader_writer(std::shared_ptr<details::reader_writer_stream> stream)
			: m_stream(std::move(stream))
		{}
		writer_on_reader_writer(const writer_on_reader_writer&) = delete;
		writer_on_reader_writer(writer_on_reader_writer&& rhs)
			: m_stream(std::move(rhs.m_stream))
		{
			rhs.m_stream = nullptr;
		}
		writer_on_reader_writer& operator = (const writer_on_reader_writer&) = delete;
		writer_on_reader_writer& operator = (writer_on_reader_writer&& rhs)
		{
			std::swap(m_stream, rhs.m_stream);
			return *this;
		}
		~writer_on_reader_writer()
		{
			if (m_stream)
				m_stream->terminate();
		}
		void write_buffer(const_buffer_ref data)
		{
			m_stream->write_buffer(data);
		}
		void flush()
		{
			return m_stream->flush();
		}

	private:
		std::shared_ptr<details::reader_writer_stream> m_stream;
	};

	struct reader_writer_pair
	{
		reader_on_reader_writer reader;
		writer_on_reader_writer writer;
	};
	inline reader_writer_pair create_reader_writer_stream()
	{
		auto inner = std::make_shared<details::reader_writer_stream>();
		return{
			reader_on_reader_writer{inner},
			writer_on_reader_writer{inner}
		};
	}
}}
