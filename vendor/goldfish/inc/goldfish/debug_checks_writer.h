#pragma once

#include "debug_checks.h"
#include "array_ref.h"
#include "tags.h"

namespace goldfish { namespace debug_checks
{
	template <class error_handler, class inner> class document_writer;
	template <class error_handler, class inner> document_writer<error_handler, std::decay_t<inner>> add_write_checks_impl(::goldfish::debug_checks::container_base<error_handler>* parent, inner&& t);

	template <class error_handler, class inner> class stream_writer : private ::goldfish::debug_checks::container_base<error_handler>
	{
	public:
		stream_writer(::goldfish::debug_checks::container_base<error_handler>* parent, inner writer)
			: ::goldfish::debug_checks::container_base<error_handler>(parent)
			, m_writer(std::move(writer))
		{}
		void write_buffer(const_buffer_ref buffer)
		{
			this -> err_if_locked();
			m_writer.write_buffer(buffer);
		}
		auto flush()
		{
			this -> err_if_locked();
			this -> unlock_parent_and_lock_self();
			return m_writer.flush();
		}
	private:
		inner m_writer;
	};
	template <class error_handler, class inner> stream_writer<error_handler, std::decay_t<inner>> add_write_checks_on_stream(::goldfish::debug_checks::container_base<error_handler>* parent, inner&& w) { return{ parent, std::forward<inner>(w) }; }

	template <class error_handler, class inner> class check_size_of_stream_writer
	{
	public:
		check_size_of_stream_writer(inner writer, uint64_t cb)
			: m_writer(std::move(writer))
			, m_cb_left(cb)
		{}
		void write_buffer(const_buffer_ref buffer)
		{
			if (m_cb_left < buffer.size())
				error_handler::on_error();

			m_writer.write_buffer(buffer);
			m_cb_left -= buffer.size();
		}
		auto flush()
		{
			if (m_cb_left != 0)
				error_handler::on_error();

			return m_writer.flush();
		}
	private:
		inner m_writer;
		uint64_t m_cb_left;
	};
	template <class error_handler, class inner> check_size_of_stream_writer<error_handler, std::decay_t<inner>> check_size_of_stream(inner&& w, uint64_t cb) { return{ std::forward<inner>(w), cb }; }

	template <class error_handler, class inner> class array_writer : private ::goldfish::debug_checks::container_base<error_handler>
	{
	public:
		array_writer(::goldfish::debug_checks::container_base<error_handler>* parent, inner writer)
			: ::goldfish::debug_checks::container_base<error_handler>(parent)
			, m_writer(std::move(writer))
		{}

		auto append()
		{
			this -> err_if_locked();
			return add_write_checks_impl(this, m_writer.append());
		}
		auto flush()
		{
			this -> err_if_locked();
			this -> unlock_parent_and_lock_self();
			return m_writer.flush();
		}
	private:
		inner m_writer;
	};
	template <class error_handler, class inner> array_writer<error_handler, std::decay_t<inner>> add_write_checks_on_array(::goldfish::debug_checks::container_base<error_handler>* parent, inner&& w) { return{ parent, std::forward<inner>(w) }; }

	template <class error_handler, class inner> class check_size_of_array_writer
	{
	public:
		check_size_of_array_writer(inner writer, uint64_t c)
			: m_writer(std::move(writer))
			, m_count_left(c)
		{}

		auto append()
		{
			if (m_count_left == 0)
				error_handler::on_error();
			--m_count_left;
			return m_writer.append();
		}
		auto flush()
		{
			if (m_count_left != 0)
				error_handler::on_error();
			return m_writer.flush();
		}
	private:
		inner m_writer;
		uint64_t m_count_left;
	};
	template <class error_handler, class inner> check_size_of_array_writer<error_handler, std::decay_t<inner>> check_size_of_array(inner&& w, uint64_t c) { return{ std::forward<inner>(w), c }; }

	template <class error_handler, class inner> class map_writer : private ::goldfish::debug_checks::container_base<error_handler>
	{
	public:
		map_writer(::goldfish::debug_checks::container_base<error_handler>* parent, inner writer)
			: ::goldfish::debug_checks::container_base<error_handler>(parent)
			, m_writer(std::move(writer))
		{}

		auto append_key()
		{
			this -> err_if_locked();
			this -> err_if_flag_set();
			this -> set_flag();
			return add_write_checks_impl(this, m_writer.append_key());
		}
		auto append_value()
		{
			this -> err_if_locked();
			this -> err_if_flag_not_set();
			this -> clear_flag();
			return add_write_checks_impl(this, m_writer.append_value());
		}
		auto flush()
		{
			this -> err_if_locked();
			this -> err_if_flag_set();
			this -> unlock_parent_and_lock_self();
			return m_writer.flush();
		}
	private:
		inner m_writer;
	};
	template <class error_handler, class inner> map_writer<error_handler, std::decay_t<inner>> add_write_checks_on_map(::goldfish::debug_checks::container_base<error_handler>* parent, inner&& w) { return{ parent, std::forward<inner>(w) }; }

	template <class error_handler, class inner> class check_size_of_map_writer
	{
	public:
		check_size_of_map_writer(inner writer, uint64_t c)
			: m_writer(std::move(writer))
			, m_c_left(c)
		{}

		auto append_key()
		{
			if (m_c_left == 0)
				error_handler::on_error();
			--m_c_left;
			return m_writer.append_key();
		}
		auto append_value()
		{
			return m_writer.append_value();
		}
		auto flush()
		{
			if (m_c_left != 0)
				error_handler::on_error();
			return m_writer.flush();
		}
	private:
		inner m_writer;
		uint64_t m_c_left;
	};
	template <class error_handler, class inner> check_size_of_map_writer<error_handler, std::decay_t<inner>> check_size_of_map(inner&& w, uint64_t c) { return{ std::forward<inner>(w), c }; }

	template <class error_handler, class inner> class document_writer : private ::goldfish::debug_checks::container_base<error_handler>
	{
	public:
		document_writer(::goldfish::debug_checks::container_base<error_handler>* parent, inner writer)
			: ::goldfish::debug_checks::container_base<error_handler>(parent)
			, m_writer(std::move(writer))
		{}
		template <class T> auto write(T&& t)
		{
			this -> err_if_locked();
			this -> unlock_parent_and_lock_self();
			return m_writer.write(std::forward<T>(t));
		}

		auto start_binary(uint64_t cb)
		{
			this -> err_if_locked();
			auto result = check_size_of_stream<error_handler>(add_write_checks_on_stream(parent(), m_writer.start_binary(cb)), cb);
			this -> lock();
			return result;
		}
		auto start_string(uint64_t cb)
		{
			this -> err_if_locked();
			auto result = check_size_of_stream<error_handler>(add_write_checks_on_stream(parent(), m_writer.start_string(cb)), cb);
			this -> lock();
			return result;
		}
		auto start_binary()
		{
			this -> err_if_locked();
			auto result = add_write_checks_on_stream(parent(), m_writer.start_binary());
			this -> lock();
			return result;
		}
		auto start_string()
		{
			this -> err_if_locked();
			auto result = add_write_checks_on_stream(parent(), m_writer.start_string());
			this -> lock();
			return result;
		}

		auto start_array(uint64_t size)
		{
			this -> err_if_locked();
			auto result = check_size_of_array<error_handler>(add_write_checks_on_array(parent(), m_writer.start_array(size)), size);
			this -> lock();
			return result;
		}
		auto start_array()
		{
			this -> err_if_locked();
			auto result = add_write_checks_on_array(parent(), m_writer.start_array());
			this -> lock();
			return result;
		}

		auto start_map(uint64_t size)
		{
			this -> err_if_locked();
			auto result = check_size_of_map<error_handler>(add_write_checks_on_map(parent(), m_writer.start_map(size)), size);
			this -> lock();
			return result;
		}
		auto start_map()
		{
			this -> err_if_locked();
			auto result = add_write_checks_on_map(parent(), m_writer.start_map());
			this -> lock();
			return result;
		}
	private:
		inner m_writer;
	};

	template <class error_handler, class inner> document_writer<error_handler, std::decay_t<inner>> add_write_checks_impl(::goldfish::debug_checks::container_base<error_handler>* parent, inner&& t)
	{
		return{ parent, std::forward<inner>(t) };
	}

	template <class error_handler, class Document> auto add_write_checks(Document&& t, error_handler)
	{
		return add_write_checks_impl<error_handler>(nullptr /*parent*/, std::forward<Document>(t));
	}
	template <class Document> auto add_write_checks(Document&& t, ::goldfish::debug_checks::no_check)
	{
		return std::forward<Document>(t);
	}
}}