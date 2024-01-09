#include "common.h"
#include "assign.h"
#include "capnp.h"


namespace fscpy {

namespace {
	struct PythonVisitor : public textio::Visitor {
		struct Uninitialized {};
		
		using NewList = py::list;
		struct PresetList {
			py::list list;
			size_t offset = 0;
		};
		struct Dict {
			py::dict dict;
			kj::Maybe<py::object> key = nullptr;
		};
		struct Forward {
			Own<textio::Visitor>;
			py::object original;
		};
		struct Done { py::object result };
		
		using State = kj::OneOf<Uninitialized, Preset, List, Dict, Forward, Done>;
		kj::Vector<State> states;
		
		State& state() { return states.back(); }
		
		void pop() {
			if(states.size() > 1)
				states.removeLast();
			else {
				KJ_REQUIRE(!state().is<Uninitialized>());
				KJ_REQUIRE(!state().is<Done>());
				
				if(state().is<NewList>()) {
					state() = Done { state().get<NewList>() };
				} else if(state().is<PresetList>()) {
					states.add(Done {p.as<PresetList>().list});
				} else if(state().is<Dict>()) {
					states.add(Done {p.as<Dict>().dict});
				} else if(state().is<Forward>()) {
					states.add(Done {p.as<Forward>().original});
				}
			}
		}
				
		#define ACCEPT_FWD(expr) \
			KJ_REQUIRE(!state().is<Done>()); \
			\
			if(state().is<Forward>()) { \
				auto& to = state().get<Forward>(); \
				to -> expr; \
				if(to -> done()) { \
					pop(); \
				} \
				return; \
			}
				
		void beginObject(Maybe<size_t> s) override {
			ACCEPT_FWD(beginObject(s));
			
			auto checkObject = [&](py::object o) {
				if(py::isinstance<DynamicStructCommon>(o)) {
					states.add(Forward {createVisitor(o), o});
				} else {
					KJ_REQUIRE(py::isinstance<py::dict>(o));
					states.add(Dict {o});
				}
			};
			

			if(state().is<Uninitialized>()) {
				state().init<Dict()>();
			} else if(state().is<NewList>()) {
				py::dict newDict;
				state().get<NewList>().append(newDict);
				states.add(Dict { newDict });
			} else if(state().is<PresetList>()) {
				auto& p = state().get<PresetList>();
				KJ_REQUIRE(p.list.size() > p.offset, "List to small to add to");
				
				py::object entry = p.list[p.offset];
				if(entry.is_none()) {
					py::dict newDict;
					states().add(Dict {newDict});
					
					p.list[p.offset] = newDict;
				} else {
					checkObject(entry);
				}
				++offset;
			} else if(state().is<Dict>()) {
				auto& dict = state().get<Dict>();
				KJ_IF_MAYBE(pKey, dict.key) {
					py::object key = mv(*pKey);
					dict.key = nullptr;
					
					if(dict.dict.contains(key)) {
						checkObject(dict.dict[key]);
					} else {
						py::dict newDict;
						states.add(Dict {newDict});
						dict.dict[key] = newDict;
					}
				} else {
					KJ_FAIL_REQUIRE("Map key must be int, float, or str, not map");
				}
			}
		}
		
		void endObject() override {
			ACCEPT_FWD(endObject())
			pop();
		}
		
		void beginArray(Maybe<size_t> s) override {
			ACCEPT_FWD(beginArray(s));
			
			auto checkList = [&](py::object o) {
				KJ_REQUIRE(py::isinstance<py::list>(o));
				states.add(Dict {o});
			};
			
			if(state().is<Uninitialized>()) {
				state().init<NewList>();
			} else if(state().is<NewList>()) {
				py::list newList;
				state().get<NewList>().append(newList);
				states.add(newList);
			} else if(state().is<PresetList>()) {
				auto& p = state().get<PresetList>();
				KJ_REQUIRE(p.list.size() > p.offset, "List to small to add to");
				
				py::object entry = p.list[p.offset];
				if(entry.is_none()) {
					py::list newList;
					states().add(newList);
					p.list[p.offset] = newList;
				} else {
					checkList(entry);
				}
				++offset;
			} else if(state().is<Dict>()) {
				auto& dict = state().get<Dict>();
				KJ_IF_MAYBE(pKey, dict.key) {
					py::object key = mv(*pKey);
					dict.key = nullptr;
					
					if(dict.dict.contains(key)) {
						checkList(dict.dict[key]);
					} else {
						py::list newList;
						states().add(newList);
						dict.dict[key] = newList;
					}
				} else {
					KJ_FAIL_REQUIRE("Map key must be int, float, or str, not list");
				}
			}
		}
		
		void endArray() override {
			ACCEPT_FWD(endArray())
			
			if(state().is<PresetList>()) {
				auto& p = state().get<PresetList>();
				KJ_REQUIRE(p.offset == p.list.size(), "Size of list loaded is smaller than template");
			}
			pop();
		}
		
		template<typename T>
		void acceptPrimitive(T t) {
			KJ_REQUIRE(!state().is<Done>());
			KJ_REQUIRE(!state().is<Forward>());
			KJ_REQUIRE(!state().is<Uninitialized>(), "Can not store primitive values in root");
			
			auto asPy = py::cast(t);
			
			if(state().is<PresetList>()) {
				auto& p = state.get<PresetList>();
				p.list[p.offset++] = asPy;
			} else if(state().is<NewList>()) {
				state.get<NewList>().append(asPy);
			} else if(state().is<Dict>()) {
				auto& d = state.get<Dict>();
				KJ_IF_MAYBE(pKey, d.key) {
					KJ_REQUIRE(!d.dict.contains(*pKey), "Dict already contains key");
					
					d.dict[*pKey] = asPy;
					d.key = nullptr;
				} else {
					d.key = asPy;
				}
			}
		}
		
		void acceptInt(int64_t i) override {
			ACCEPT_FWD(acceptInt(i));
			acceptPrimitive(i);
		}
		
		void acceptUInt(uint64_t i) override {
			ACCEPT_FWD(acceptUInt(i));
			acceptPrimitive(i);
		}
		
		void acceptDouble(double d) override {
			ACCEPT_FWD(acceptDouble(d));
			acceptPrimitive(d);
		}
		
		void acceptString(kj::StringPtr s) override {
			ACCEPT_FWD(acceptUInt(s));
			acceptPrimitive(s);
		}
		
		void acceptData(kj::ArrayPtr<const kj::byte> d) override {
			ACCEPT_FWD(acceptData(d));
			acceptPrimitive(py::bytes(d.begin(), d.size()));
		}
		
		PythonVisitor(py::object o) {
			if(o.is_none()) {
				states.add(Uninitialized());
			} else if(py::isinstance<py::list>()) {
				states.add(PresetList { py::cast<py::list>(o) });
			} else if(py::isinstance<py::dict>()) {
				states.add(Dict { py::cast<py::dict>(o) });
			} else if(py::isinstance<DynamicStructCommon>(o)) {
				auto& ds = py::cast<DynamicStructCommon&>(o);
				states.add(Forward {createVisitor(ds), o});
			} else {
				KJ_FAIL_REQUIRE("Can not read into object of specfied type. Must be None, list, dict, or a struct builder");
			}
		}

	};		
}

namespace formats {
	void Format::dump(DynamicValueReader r, py::object o, bool compact) {
		int fd = o.attr("fileno")();
		kj::FdOutputStream os(fd);
		
		// Flush python-side buffers before writing
		o.attr("flush")();
		
		// Prepare write options
		textio::WriteOptions wopts {
			.compact = compact
		};
		
		write(r, os, wopts);
	}
	
	Formatted Format::load(py::object o) {
		int fd = o.attr("fileno")();
		return Formatted { *this, kj::heap<kj::FdInputStream>(fd), o };
	}
	
	void Format::dumps(DynamicValueReader r, bool compact) {
		// Prepare write options
		textio::WriteOptions wopts {
			.compact = compact
		};
		
		kj::VectorOutputStream os;
		write(r, os, wopts);
		
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
	
	TextIOFormat::TextIOFormat(const textio::Dialect& d, const textio::WriteOptions& wo) :
		dialect(d), writeOptions(wo)
	{}
	
	void TextIOFormat::write(DynamicValueReader r, kj::BufferedOutputStream& os, const textio::WriteOptions& writeOptions) {
		textio::save(r, *textio::createVisitor(os), dialect, writeOptions);
	}
	
	void TextIOFormat::read(const BuilderSlot& slot, kj::BufferedInputStream& bis) {		
		if(dst.type.isList()) {
			auto initializer = [&](size_t s) {
				return dst.init(s).as<capnp::DynamicList>();
			};
			textio::load(bis, *createVisitor(dst.type.asList(), mv(initializer)), dialect);
			return;
		} else if(dst.type.isStruct()) {
			auto asStruct = dst.init().as<capnp::DynamicStruct>();
			textio::load(bis, *createVisitor(asStruct), dialect);
			return;
		}
		
		KJ_FAIL_REQUIRE("Can only assign struct and list types");
	}
	
	void YAML::YAML(
	
	namespace {
		struct YAML : TextIOFormat {
			YAML(bool compact) : TextIOFormat(textio::Dialect {
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
	
	using Dialect = textio::Dialect;
	using F = TextIOFormat;
	
	static TextIOFormat yaml(Dialect{
		.language = Dialect::YAML;
	});
	static TextIOFormat json(Dialect{
		.language = Dialect::JSON;
	});
	static TextIOFormat bson(Dialect{
		.language = Dialect::BSON;
	});
	static TextIOFormat cbor(Dialect{
		.language = Dialect::CBOR;
	});
	
	static D yaml { .language = D::YAML; }
	static JsonDialect json;
	static JsonDialect cbor(JsonOptions { dialect = JsonOptions::CBOR });
	static JsonDialect bson(JsonOptions { dialect = JsonOptions::BSON });
	
	formatsMod.attr("yaml") = py::cast<Format&>(yaml);
	formatsMod.attr("json") = py::cast<Format&>(json);
	formatsMod.attr("cbor") = py::cast<Format&>(cbor);
	formatsMod.attr("bson") = py::cast<Format&>(bson);
}

}