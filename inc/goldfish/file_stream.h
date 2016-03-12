#pragma once

#include "array_ref.h"
#include <string>

namespace goldfish { namespace stream
{
	class file_reader
	{
	public:
		file_reader(const char* path)
		{
			if (auto error = fopen_s(&m_fp, path, "rb"))
				throw error;
		}
		file_reader(const std::string& path)
			: file_reader(path.c_str())
		{}
		file_reader(file_reader&& rhs)
			: m_fp(rhs.m_fp)
		{
			rhs.m_fp = nullptr;
		}
		~file_reader()
		{
			if (m_fp)
				fclose(m_fp);
		}
		file_reader(const file_reader&) = delete;
		file_reader& operator = (const file_reader&) = delete;

		size_t read_buffer(buffer_ref data)
		{
			return fread(data.data(), 1, data.size(), m_fp);
		}
	private:
		FILE* m_fp;
	};

	class file_writer
	{
	public:
		file_writer(const char* path)
		{
			if (auto error = fopen_s(&m_fp, path, "wb"))
				throw error;
		}
		file_writer(file_writer&& rhs)
			: m_fp(rhs.m_fp)
		{
			rhs.m_fp = nullptr;
		}
		~file_writer()
		{
			if (m_fp)
				fclose(m_fp);
		}
		file_writer(const file_writer&) = delete;
		file_writer& operator = (const file_writer&) = delete;

		void write_buffer(const_buffer_ref data)
		{
			if (fwrite(data.data(), 1, data.size(), m_fp) != data.size())
				throw 0;
		}
		void flush() { }
	private:
		FILE* m_fp;
	};
}}