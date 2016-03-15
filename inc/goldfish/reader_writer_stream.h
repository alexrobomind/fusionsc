#pragma once

#include "array_ref.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace goldfish { namespace stream
{
	/* This acts in a similar manner to a producer consumer queue */
	class reader_writer_stream
	{
	public:
		size_t read_buffer(buffer_ref data)
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_read_buffer_to_be_filled = data;
			m_condition_variable.notify_one(); // Wake up the writer (now that m_read_buffer_to_be_filled is likely not empty)
			m_condition_variable.wait(lock, [&] { return m_is_flushed || m_read_buffer_to_be_filled.empty(); });
			return data.size() - m_read_buffer_to_be_filled.size();
		}

		void write_buffer(const_buffer_ref data)
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			assert(!m_is_flushed);

			while (!data.empty())
			{
				m_condition_variable.wait(lock, [&] { return !m_read_buffer_to_be_filled.empty(); });
				auto to_copy = std::min(m_read_buffer_to_be_filled.size(), data.size());
				copy(data.remove_front(to_copy), m_read_buffer_to_be_filled.remove_front(to_copy));
				if (m_read_buffer_to_be_filled.empty())
					m_condition_variable.notify_one();
			}
		}
		void flush()
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			assert(!m_is_flushed);
			m_is_flushed = true;
			m_condition_variable.notify_one();
		}

	private:
		std::mutex m_mutex;
		std::condition_variable m_condition_variable;
		buffer_ref m_read_buffer_to_be_filled;
		bool m_is_flushed = false;
	};
}}
