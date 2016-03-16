#pragma once

#include "dom.h"

namespace goldfish
{
	namespace details
	{
		template <class writer> void write(writer& writer, const dom::document& d)
		{
			d.visit([&](auto&& x)
			{ 
				details::write(writer, x);
			});
		}

		template <class writer> void write(writer& writer, bool x) { writer.write(x); }
		template <class writer> void write(writer& writer, nullptr_t x) { writer.write(x); }
		template <class writer> void write(writer& writer, tags::undefined x) { writer.write(x); }
		template <class writer> void write(writer& writer, uint64_t x) { writer.write(x); }
		template <class writer> void write(writer& writer, int64_t x) { writer.write(x); }
		template <class writer> void write(writer& writer, double x) { writer.write(x); }

		template <class writer> void write(writer& writer, const std::vector<uint8_t>& x)
		{
			auto d = writer.start_binary(x.size());
			d.write_buffer(const_buffer_ref{ x.data(), x.size() });
			d.flush();
		}
		template <class writer> void write(writer& writer, const std::string& x)
		{
			auto d = writer.start_string(x.size());
			d.write_buffer(const_buffer_ref{ reinterpret_cast<const uint8_t*>(x.data()), x.size() });
			d.flush();
		}
		template <class writer> void write(writer& writer, const dom::array& xs)
		{
			auto d = writer.start_array(xs.size());
			for (auto&& x : xs)
				copy_dom_document(d.append(), x);
			d.flush();
		}
		template <class writer> void write(writer& writer, const dom::map& xs)
		{
			auto d = writer.start_map(xs.size());
			for (auto&& x : xs)
			{
				copy_dom_document(d.append_key(), x.first);
				copy_dom_document(d.append_value(), x.second);
			}
			d.flush();
		}
	}

	template <class writer> void copy_dom_document(writer&& w, const dom::document& d)
	{
		details::write(w, d);
	}
}