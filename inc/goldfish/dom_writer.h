#pragma once

#include "dom.h"

namespace goldfish { namespace dom
{
	namespace details
	{
		template <class writer> void write(writer& writer, const document& d)
		{
			d.visit([&](auto&& x)
			{ 
				details::write(writer, x);
			});
		}

		template <class writer> void write(writer& writer, bool x) { writer.write(x); }
		template <class writer> void write(writer& writer, nullptr_t) { writer.write(nullptr); }
		template <class writer> void write(writer& writer, tags::undefined) { writer.write_undefined(); }
		template <class writer> void write(writer& writer, uint64_t x) { writer.write(x); }
		template <class writer> void write(writer& writer, int64_t x) { writer.write(x); }
		template <class writer> void write(writer& writer, double x) { writer.write(x); }

		template <class writer> void write(writer& writer, const std::vector<uint8_t>& x)
		{
			auto d = writer.write_binary(x.size());
			d.write_buffer(const_buffer_ref{ x.data(), x.size() });
			d.flush();
		}
		template <class writer> void write(writer& writer, const std::string& x)
		{
			auto d = writer.write_text(x.size());
			d.write_buffer(const_buffer_ref{ reinterpret_cast<const uint8_t*>(x.data()), x.size() });
			d.flush();
		}
		template <class writer> void write(writer& writer, const array& xs)
		{
			auto d = writer.write_array(xs.size());
			for (auto&& x : xs)
				write(d.append(), x);
			d.flush();
		}
		template <class writer> void write(writer& writer, const map& xs)
		{
			auto d = writer.write_map(xs.size());
			for (auto&& x : xs)
			{
				write(d.append_key(), x.first);
				write(d.append_value(), x.second);
			}
			d.flush();
		}
	}

	template <class writer> void write(writer&& w, const document& d)
	{
		details::write(w, d);
	}
}}