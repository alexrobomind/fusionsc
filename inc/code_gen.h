#pragma once

#include "dom_reader.h"
#include "json_reader.h"

/*
	Input is a JSON document that defines the different types
	We have the following built in types:
		- "binary": a binary stream (fixed sized or not)
		- "text": a UTF8 string (fixed sized or not)
		- "float": a floating point number
		- "uint8", "uint16", "uint32", "uint64": unsigned integers
		- "int8", "int16", "int32", "int64": signed integers
		- "bool": a boolean value (true or false)
		- "document": any data
		- [type]: an array containing items of type "type"
		- {"key":type_key, "value":type_value}: a map of type_key objects to type_value objects

	The input is a JSON document that defines additional types in the following format:
		{
			"AliasName":type, // generates an alias

			"TypeName":{
				"FieldName":{
					"type":type,
					"optional":true/false,
				}, ...
			}, ...
		}

	The type field is mandatory, it indicates the expected type of the "FieldName" object
	The tag field is optional. It's an integer used in the binary protocols like CBOR in place of the field name for concision and performance.
		If unspecified, the serialization uses the string version
	The optional field is optional and defaults to false. If true, the field might be ommitted from the data.
*/

namespace gold_fish {
	struct bad_type_declaration {};

	namespace details
	{
		struct parsed_field
		{
			std::string name;
			std::string type;
			bool optional;
		};
		std::string parse_type(const dom::document& d);

		inline parsed_field parse_field(std::string name, const dom::document& d)
		{
			if (d.is<std::string>())
			{
				return{ std::move(name), d.as<std::string>(), false /*optional*/ };
			}
			else if (d.is<dom::map>())
			{
				auto&& fields = d.as<dom::map>();
				bool optional = std::count_if(fields.begin(), fields.end(), [](auto&& kv)
				{
					return kv.first.as<std::string>() == "optional" && kv.second.as<bool>();
				}) != 0;
				auto it_type = std::find_if(fields.begin(), fields.end(), [](auto&& kv)
				{
					return kv.first.as<std::string>() == "type";
				});
				if (it_type == fields.end())
					throw bad_type_declaration{};
				return{ std::move(name), parse_type(it_type->second.as<std::string>()), optional };
			}
			else
			{
				throw bad_type_declaration{};
			}
		}

		template <class OutputStream, class Map>
		void generate_code_for_type(OutputStream& output, const std::string& type_name, Map& stream_fields)
		{
			std::vector<parsed_field> fields;
			while (auto key = stream_fields.read_key())
			{
				auto name = read_all_as_string(key->as<tags::text_string>());
				fields.emplace_back(parse_field(std::move(name), dom::load_in_memory(stream_fields.read_value())));
			}

			size_t max_size = 0;
			for (auto&& field : fields)
				max_size = std::max(max_size, field.name.size());
			output
				<< "template <class Map> class " << type_name << "\n"
				<< "{\n"
				<< "public:\n"
				<< "	" << type_name << "(Map map)\n"
				<< "		: m_map(std::move(map))\n"
				<< "	{}\n";

			for (auto it = fields.begin(), end = fields.end(); it != end; ++it)
			{
				auto&& field = *it;
				auto field_index = std::distance(fields.begin(), it);

				if (field.optional)
					output << "	auto " << field.name << "() -> optional<std::decay_t<decltype(" << field.type << "::cast(std::declval<Map>().read_value()))>>\n";
				else
					output << "	auto " << field.name << "()\n";

				output
					<< "	{\n"
					<< "		uint8_t buffer[" << max_size + 1 << "];\n"
					<< "		if (!m_map.try_move_to(field_names(), buffer, " << field_index << ", " << field_index + 1 << "))\n";

				if (field.optional)
					output << "			return nullopt;\n";
				else
					output << "			throw bad_type_declaration{};\n";

				output
					<< "		return " << field.type << "::cast(m_map.read_value());\n"
					<< "	}\n";
			}

			output
				<< "private:\n"
				<< "	static auto field_names()\n"
				<< "	{\n"
				<< "		static const array_ref<const char> result[] = {\n";

			for (auto&& field : fields)
				output << "			array_ref<const char>(\"" << field.name << "\").without_end(1),\n";

			output
				<< "		};\n"
				<< "		return make_array_ref(result);\n"
				<< "	}\n"
				<< "	filtered_map<Map> m_map;\n"
				<< "};\n";
		}
	}

	template <class Stream, class OutputStream>
	void generate_code(OutputStream& output, Stream s)
	{
		auto x = json::read(std::move(s));
		if (!x.is<tags::map>())
			throw bad_type_declaration();
		auto& list_of_types = x.as<tags::map>();
		while (auto key = list_of_types.read_key())
		{
			if (!key->is<tags::text_string>())
				throw bad_type_declaration();

			auto name = stream::read_all_as_string(key->as<tags::text_string>());
			auto value = list_of_types.read_value();
			if (!value.is<tags::map>())
				throw bad_type_declaration();
			details::generate_code_for_type(output, name, value.as<tags::map>());
		}
	}
}
