#incldyue "common.h"
#include "assign.h"


namespace fscpy {

namespace formats {
	void Format::dump(DynamicValueReader r, py::object o) {
		int fd = o.attr("fileno")();
		kj::FdOutputStream os(fd);
		
		// Flush python-side buffers before writing
		o.attr("flush")();
		
		write(r, os);
	}
	
	Formatted Format::load(py::object o) {
		int fd = o.attr("fileno")();
		return Formatted { *this, kj::heap<kj::FdInputStream>(fd), o };
	}
	
	void Format::dumps(DynamicValueReader r) {
		kj::VectorOutputStream os;
		write(r, os);
		
		auto arr = os.getArray();
		
		if(isBinary) {
			return py::bytes(arr.begin(), arr.size());
		} else {
			return py::str(arr.begin(), arr.size());
		}
	}
	
	Formatted Format::loads1(py::buffer buf) {
		py::buffer_info info = buf.request();
		
		KJ_REQUIRE(info.itemsize == 1, "Can only read from character buffers");
		
		return Formatted {
			*this,
			kj::heap<kj::ArrayInputStream>(
				kj::ArrayPtr<const kj::byte>(info.ptr, info.size)
			),
			buf
		};
	}
	
	Formatted Format::loads2(py::str str) {		
		return loads1(
			py::reinterpret_steal<py::buffer>(
				PyUnicode_AsUTF8String(str)
			)
		);
	}
		
	py::object Format::get(DynamicValueReader reader) {
		VectorOutputStream os;
		write(reader, os);
		
		auto arr = os.getArray();
	}
	
	void YAML::write(DynamicValueReader reader, kj::BufferedOutputStream& os) {
		Own<std::ostream> wrapped = toStdStream(os);		
		YAML::Emitter document(*wrapped);

		document << reader;
		
		wrapped -> flush();
		os.flush();
	}
	
	void YAML::read(const BuilderSlot& dst, kj::BufferedInputStream& is) {
		Own<std::istream> wrapped = toStdStream(is);		
		YAML::Node node = YAML::Load(*wrapped);
		
		if(dst.type.isList()) {
			auto asList = dst.init(node.size()).as<capnp::DynamicList>();
			load(asList, node);
			return;
		} else if(dst.type.isStruct()) {
			auto asStruct = dst.init().as<capnp::DynamicStruct>();
			load(asStruct, node);
			return;
		}
		
		KJ_FAIL_REQUIRE("Can only assign struct and list types from YAML");
	}
	
	void JsonDialect::write(DynamicValueReader reader, kj::BufferedOutputStream& os) {
		writeJson(reader, os, opts);
	}
	
	void JsonDialect::read(const BuilderSlot& dst, kj::BufferedInputStream& is) {
		if(dst.type.isList()) {
			auto initializer = [&](size_t s) {
				return dst.init(s).as<capnp::DynamicList>();
			};
			
			loadJson(dst.type.asList(), initializer, is, node);
			
			return;
		} else if(dst.type.isStruct()) {
			auto asStruct = dst.init().as<capnp::DynamicStruct>();
			
			loadJson(asStruct, is, opts);
			
			return;
		}
		
		KJ_FAIL_REQUIRE("Can only assign struct and list types from JSON / BSON / CBOR");
		
	}
}

void initFormats(py::module_& m) {
	auto formatsMod = m.def_submodule("formats");
	
	py::class_<Formatted, Assignable>(
		"Formatted",
		"A format-aware data stream. The object can not access the data directly,"
		" but can be assigned to a struct / list builder or used as an argument to"
		" StructType.newMessage(...)."
	);
	
	py::class_<Format>("Format")
		.def_readonly(&Format::isBinary, "isBinary")
		
		.def(&Format::load, "load")
		.def(&Format::loads1, "loads")
		.def(&Format::loads2, "loads")
		
		.def(&Format::dump, "dump")
		.def(&Format::dumps, "dumps")
	;
	
	static YAML yaml;
	static JsonDialect json;
	static JsonDialect cbor(JsonOptions { dialect = JsonOptions::CBOR });
	static JsonDialect bson(JsonOptions { dialect = JsonOptions::BSON });
	
	formatsMod.attr("yaml") = py::cast<Format&>(yaml);
	formatsMod.attr("json") = py::cast<Format&>(json);
	formatsMod.attr("cbor") = py::cast<Format&>(cbor);
	formatsMod.attr("bson") = py::cast<Format&>(bson);
}

}