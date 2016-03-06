#pragma once

#include "uncaught_exception.h"

#ifndef NDEBUG
namespace gold_fish { namespace debug_check
{
	template <class inner> class document_writer;
	template <class inner> document_writer<std::decay_t<inner>> add_write_checks(inner&& t);

	template <class inner> class stream_writer : private assert_work_done
	{
	public:
		stream_writer(inner writer)
			: m_writer(std::move(writer))
		{}
		void write_buffer(const_buffer_ref buffer)
		{
			assert(!is_work_done());
			m_writer.write_buffer(buffer);
		}
		void flush()
		{
			assert(!is_work_done());
			m_writer.flush();
			mark_work_done();
		}
	private:
		inner m_writer;
	};
	template <class inner> stream_writer<std::decay_t<inner>> add_write_checks_on_stream(inner&& w) { return{ std::forward<inner>(w) }; }

	template <class inner> class check_size_of_stream_writer
	{
	public:
		check_size_of_stream_writer(inner writer, uint64_t cb)
			: m_writer(std::move(writer))
			, m_cb_left(cb)
		{}
		void write_buffer(const_buffer_ref buffer)
		{
			m_writer.write_buffer(buffer);
			m_cb_left -= buffer.size();
		}
		void flush()
		{
			assert(m_cb_left == 0);
			m_writer.flush();
		}
	private:
		inner m_writer;
		uint64_t m_cb_left;
	};
	template <class inner> check_size_of_stream_writer<std::decay_t<inner>> check_size_of_stream(inner&& w, uint64_t cb) { return{ std::forward<inner>(w), cb }; }

	template <class inner> class array_writer : private assert_work_done
	{
	public:
		array_writer(inner writer)
			: m_writer(std::move(writer))
		{}

		auto append()
		{
			assert(!is_work_done());
			return add_write_checks(m_writer.append());
		}
		void flush()
		{
			assert(!is_work_done());
			m_writer.flush();
			mark_work_done();
		}
	private:
		inner m_writer;
	};
	template <class inner> array_writer<std::decay_t<inner>> add_write_checks_on_array(inner&& w) { return{ std::forward<inner>(w) }; }

	template <class inner> class check_size_of_array_writer
	{
	public:
		check_size_of_array_writer(inner writer, uint64_t c)
			: m_writer(std::move(writer))
			, m_c_left(c)
		{}

		auto append()
		{
			assert(m_c_left > 0);
			--m_c_left;
			return add_write_checks(m_writer.append());
		}
		void flush()
		{
			assert(m_c_left == 0);
			m_writer.flush();
		}
	private:
		inner m_writer;
		uint64_t m_c_left;
	};
	template <class inner> check_size_of_array_writer<std::decay_t<inner>> check_size_of_array(inner&& w, uint64_t c) { return{ std::forward<inner>(w), c }; }

	template <class inner> class map_writer : private assert_work_done
	{
	public:
		map_writer(inner writer)
			: m_writer(std::move(writer))
		{}

		auto append_key() 
		{
			assert(!is_work_done());
			assert(m_expect_key);
			m_expect_key = false;
			return add_write_checks(m_writer.append_key());
		}
		auto append_value()
		{
			assert(!is_work_done());
			assert(!m_expect_key);
			m_expect_key = true;
			return add_write_checks(m_writer.append_value());
		}
		void flush()
		{
			assert(!is_work_done());
			assert(m_expect_key);
			m_writer.flush();
			mark_work_done();
		}
	private:
		inner m_writer;
		bool m_expect_key = true;
	};
	template <class inner> map_writer<std::decay_t<inner>> add_write_checks_on_map(inner&& w) { return{ std::forward<inner>(w) }; }

	template <class inner> class check_size_of_map_writer
	{
	public:
		check_size_of_map_writer(inner writer, uint64_t c)
			: m_writer(std::move(writer))
			, m_c_left(c)
		{}

		auto append_key()
		{
			assert(m_c_left > 0);
			--m_c_left;
			return add_write_checks(m_writer.append_key());
		}
		auto append_value()
		{
			return add_write_checks(m_writer.append_value());
		}
		void flush()
		{
			assert(m_c_left == 0);
			m_writer.flush();
		}
	private:
		inner m_writer;
		uint64_t m_c_left;
	};
	template <class inner> check_size_of_map_writer<std::decay_t<inner>> check_size_of_map(inner&& w, uint64_t c) { return{ std::forward<inner>(w), c }; }


	template <class inner> class document_writer
	{
	public:
		document_writer(inner writer)
			: m_writer(std::move(writer))
		{}
		void write(bool x) { m_writer.write(x); }
		void write(nullptr_t x) { m_writer.write(x); }
		void write(double x) { m_writer.write(x); }
		void write_undefined() { m_writer.write_undefined(); }
		void write(uint64_t x) { m_writer.write(x); }
		void write(int64_t x) { m_writer.write(x); }
		void write(uint32_t x) { m_writer.write(x); }
		void write(int32_t x) { m_writer.write(x); }

		auto write_binary(uint64_t cb) { return check_size_of_stream(add_write_checks_on_stream(m_writer.write_binary(cb)), cb); }
		auto write_text(uint64_t cb) { return check_size_of_stream(add_write_checks_on_stream(m_writer.write_text(cb)), cb); }
		auto write_binary() { return add_write_checks_on_stream(m_writer.write_binary()); }
		auto write_text() { return add_write_checks_on_stream(m_writer.write_text()); }

		auto write_array(uint64_t size) { return check_size_of_array(add_write_checks_on_array(m_writer.write_array(size)), size); }
		auto write_array() { return add_write_checks_on_array(m_writer.write_array()); }

		auto write_map(uint64_t size) { return check_size_of_map(add_write_checks_on_map(m_writer.write_map(size)), size); }
		auto write_map() { return add_write_checks_on_map(m_writer.write_map()); }

	private:
		inner m_writer;
	};

	template <class inner> document_writer<std::decay_t<inner>> add_write_checks(inner&& t)
	{
		return std::forward<inner>(t);
	}
	template <class inner> document_writer<inner> add_write_checks(document_writer<inner>&& t)
	{
		return std::move(t);
	}
	template <class inner> document_writer<inner> add_write_checks(const document_writer<inner>& t)
	{
		return t;
	}
}}
#else
namespace gold_fish { namespace debug_check
{
	template <class T> decltype(auto) add_write_checks(T&& t)
	{
		return std::forward<T>(t);
	}
}}
#endif