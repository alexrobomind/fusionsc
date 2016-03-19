#pragma once

#include "array_ref.h"
#include <string>

namespace goldfish { namespace stream
{
	struct io_exception : exception {};
	struct io_exception_with_error_code { int error_code; };

	class file_handle
	{
	public:
		file_handle(const char* path, const char* mode, const wchar_t* wmode)
		{
			if (auto error = fopen_s(&m_fp, path, mode))
				throw io_exception_with_error_code{ error };
		}
		file_handle(const wchar_t* path, const char* mode, const wchar_t* wmode)
		{
			if (auto error = _wfopen_s(&m_fp, path, wmode))
				throw io_exception_with_error_code{ error };
		}
		file_handle(const std::string& path, const char* mode, const wchar_t* wmode)
			: file_handle(path.c_str(), mode, wmode)
		{}
		file_handle(const std::wstring& path, const char* mode, const wchar_t* wmode)
			: file_handle(path.c_str(), mode, wmode)
		{}

		file_handle(file_handle&& rhs)
			: m_fp(rhs.m_fp)
		{
			rhs.m_fp = nullptr;
		}
		~file_handle()
		{
			if (m_fp)
				fclose(m_fp);
		}
		file_handle(const file_handle&) = delete;
		file_handle& operator = (const file_handle&) = delete;

		FILE* get() const { return m_fp; }
	private:
		FILE* m_fp;
	};

	class file_reader
	{
	public:
		template <class T> file_reader(T&& t)
			: m_file(std::forward<T>(t), "rb", L"rb")
		{}
		size_t read_buffer(buffer_ref data)
		{
			auto cb = fread(data.data(), 1 /*size*/, data.size() /*count*/, m_file.get());
			if (cb != data.size())
			{
				if (auto error = ferror(m_file.get()))
					throw io_exception_with_error_code{ error };
			}
			return cb;
		}
	private:
		file_handle m_file;
	};

	class file_writer
	{
	public:
		template <class T> file_writer(T&& t)
			: m_file(std::forward<T>(t), "wb", L"wb")
		{}
		void write_buffer(const_buffer_ref data)
		{
			if (fwrite(data.data(), 1 /*size*/, data.size() /*count*/, m_file.get()) != data.size())
			{
				if (auto error = ferror(m_file.get()))
					throw io_exception_with_error_code{ error };
			}
		}
		void flush() { }
	private:
		file_handle m_file;
	};
}}