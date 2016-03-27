#pragma once

#include "array_ref.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace goldfish { namespace stream
{
	struct reader_writer_stream_closed : exception {};

	namespace details
	{
		/* This acts in a similar manner to a producer consumer queue */
		class reader_writer_stream
		{
		public:
			size_t read_buffer(buffer_ref data)
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				assert(m_state != state::opened || m_read_buffer_to_be_filled.empty());
				m_read_buffer_to_be_filled = data;
				m_condition_variable.notify_one(); // Wake up the writer (now that m_read_buffer_to_be_filled is likely not empty)
				m_condition_variable.wait(lock, [&] { return m_state != state::opened || m_read_buffer_to_be_filled.empty(); });
				if (m_state == state::closed)
					throw reader_writer_stream_closed();
				return data.size() - m_read_buffer_to_be_filled.size();
			}

			void write_buffer(const_buffer_ref data)
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				assert(m_state != state::flushed);

				while (!data.empty())
				{
					m_condition_variable.wait(lock, [&] { return m_state == state::closed || !m_read_buffer_to_be_filled.empty(); });
					if (m_state == state::closed)
						throw reader_writer_stream_closed();

					auto to_copy = std::min(m_read_buffer_to_be_filled.size(), data.size());
					copy(data.remove_front(to_copy), m_read_buffer_to_be_filled.remove_front(to_copy));
					if (m_read_buffer_to_be_filled.empty())
						m_condition_variable.notify_one();
				}
			}
			void flush()
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				assert(m_state != state::flushed);
				if (m_state == state::closed)
					throw reader_writer_stream_closed{};

				m_state = state::flushed;
				m_condition_variable.notify_one();
			}
			void close()
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				if (m_state == state::opened)
				{
					m_state = state::flushed;
					m_condition_variable.notify_one();
				}
			}

		private:
			std::mutex m_mutex;
			std::condition_variable m_condition_variable;
			buffer_ref m_read_buffer_to_be_filled;

			enum class state
			{
				opened,
				flushed,
				closed,
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
		reader_on_reader_writer& operator = (reader_on_reader_writer&&) = delete;
		~reader_on_reader_writer()
		{
			if (m_stream)
				m_stream->close();
		}
		size_t read_buffer(buffer_ref data)
		{
			return m_stream->read_buffer(data);
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
		writer_on_reader_writer& operator = (writer_on_reader_writer&&) = delete;
		~writer_on_reader_writer()
		{
			if (m_stream)
				m_stream->close();
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

	inline std::pair<reader_on_reader_writer, writer_on_reader_writer> create_reader_writer_stream()
	{
		auto inner = std::make_shared<details::reader_writer_stream>();
		return{
			reader_on_reader_writer{inner},
			writer_on_reader_writer{inner}
		};
	}
}}
