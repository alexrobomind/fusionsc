#include "fscpy.h"
#include "assign.h"
#include "capnp.h"

#include "formats.h"


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
			Own<textio::Visitor> visitor;
			py::object original;
		};
		struct Done { py::object result; };
		
		using State = kj::OneOf<Uninitialized, PresetList, NewList, Dict, Forward, Done>;
		kj::Vector<State> states;
		
		State& state() { return states.back(); }
		
		bool done() override {
			return state().is<Done>();
		}
		
		void pop() {
			if(states.size() > 1)
				states.removeLast();
			else {
				auto& s = state();
				
				KJ_REQUIRE(!s.is<Uninitialized>());
				KJ_REQUIRE(!s.is<Done>());
				
				if(s.is<NewList>()) {
					s = Done { s.get<NewList>() };
				} else if(s.is<PresetList>()) {
					states.add(Done {s.get<PresetList>().list});
				} else if(s.is<Dict>()) {
					states.add(Done {s.get<Dict>().dict});
				} else if(s.is<Forward>()) {
					states.add(Done {s.get<Forward>().original});
				}
			}
		}
				
		#define ACCEPT_FWD(expr) \
			KJ_REQUIRE(!state().is<Done>()); \
			\
			if(state().is<Forward>()) { \
				auto& to = state().get<Forward>().visitor; \
				to -> expr; \
				if(to -> done()) { \
					pop(); \
				} \
				return; \
			}
				
		void beginObject(Maybe<size_t> s) override {
			ACCEPT_FWD(beginObject(s));
			
			auto checkObject = [&](py::object o) {
				if(py::isinstance<DynamicStructBuilder>(o)) {
					states.add(Forward {textio::createVisitor(py::cast<DynamicStructBuilder>(o)), o});
				} else {
					KJ_REQUIRE(py::isinstance<py::dict>(o));
					states.add(Dict {o});
				}
			};
			

			if(state().is<Uninitialized>()) {
				state().init<Dict>();
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
					states.add(Dict {newDict});
					
					p.list[p.offset] = newDict;
				} else {
					checkObject(entry);
				}
				++state().get<PresetList>().offset;
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
					states.add(newList);
					p.list[p.offset] = newList;
				} else {
					checkList(entry);
				}
				++p.offset;
			} else if(state().is<Dict>()) {
				auto& dict = state().get<Dict>();
				KJ_IF_MAYBE(pKey, dict.key) {
					py::object key = mv(*pKey);
					dict.key = nullptr;
					
					if(dict.dict.contains(key)) {
						checkList(dict.dict[key]);
					} else {
						py::list newList;
						states.add(newList);
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
				auto& p = state().get<PresetList>();
				p.list[p.offset++] = asPy;
			} else if(state().is<NewList>()) {
				state().get<NewList>().append(asPy);
			} else if(state().is<Dict>()) {
				auto& d = state().get<Dict>();
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
			ACCEPT_FWD(acceptString(s));
			acceptPrimitive(s);
		}
		
		void acceptData(kj::ArrayPtr<const kj::byte> d) override {
			ACCEPT_FWD(acceptData(d));
			acceptPrimitive(py::bytes((const char*) d.begin(), d.size()));
		}
		
		void acceptNull() override {
			ACCEPT_FWD(acceptNull());
			acceptPrimitive(py::none());
		}
		
		void acceptBool(bool b) override {
			ACCEPT_FWD(acceptBool(b))
			acceptPrimitive(b);
		}
		
		PythonVisitor(py::object o) {
			if(o.is_none()) {
				states.add(Uninitialized());
			} else if(py::isinstance<py::list>(o)) {
				states.add(PresetList { py::cast<py::list>(o) });
			} else if(py::isinstance<py::dict>(o)) {
				states.add(Dict { py::cast<py::dict>(o) });
			} else if(py::isinstance<DynamicStructBuilder>(o)) {
				auto& ds = py::cast<DynamicStructBuilder&>(o);
				states.add(Forward {textio::createVisitor(ds), o});
			} else {
				KJ_FAIL_REQUIRE("Can not read into object of specfied type. Must be None, list, dict, or a struct builder");
			}
		}

	};		
}

namespace formats {
	FormattedReader Format::open(py::object o) {
		using BufferPtr = kj::ArrayPtr<const kj::byte>;
		
		if(py::isinstance<py::str>(o)) {
			Py_ssize_t size;
			const char* utf8 = PyUnicode_AsUTF8AndSize(o.ptr(), &size);
			if(utf8 == nullptr)
				throw py::error_already_set();
			
			auto inputStream = kj::heap<kj::ArrayInputStream>(
				BufferPtr((const kj::byte*) utf8, size)
			);
			inputStream = inputStream.attach(o);
			
			return FormattedReader(*this, mv(inputStream));
		}
		
		if(py::isinstance<py::buffer>(o)) {
			auto buf = py::cast<py::buffer>(o);
			
			py::buffer_info info = buf.request();
			
			KJ_REQUIRE(info.itemsize == 1, "Can only read from character buffers");
			
			auto inputStream = kj::heap<kj::ArrayInputStream>(
				BufferPtr((const kj::byte*) info.ptr, info.size)
			);
			inputStream = inputStream.attach(mv(info));
			
			return FormattedReader(*this, mv(inputStream));
		}
		
		if(py::hasattr(o, "fileno")) {
			int fd = py::cast<int>(o.attr("fileno")());
			auto unbuffered = kj::heap<kj::FdInputStream>(fd).attach(o);
			auto buffered = kj::heap<kj::BufferedInputStreamWrapper>(*unbuffered);
			return FormattedReader(*this, buffered.attach(mv(unbuffered)));
		}
		
		KJ_FAIL_REQUIRE("Input object must be either str, file-like (with a file descriptor) or support the buffer protocol");
	}
	
	namespace {
		void dumpToVisitor(py::object src, textio::Visitor& dst);
		
		template<typename T>
		void dumpList(py::object src, textio::Visitor& dst) {
			auto asList = py::cast<T>(src);
			dst.beginArray(asList.size());
			for(auto e : asList)
				dumpToVisitor(e, dst);
			dst.endArray();
		}
		
		void dumpToVisitor(py::object src, textio::Visitor& dst) {
			// Fast non-copy path for string pointers
			py::detail::make_caster<kj::StringPtr> strCaster;
			if(strCaster.load(src, false)) {
				dst.acceptString(strCaster);
				return;
			}
			
			// Collection types
			if(py::isinstance<py::tuple>(src)) {
				dumpList<py::tuple>(src, dst);
				return;
			}
			
			if(py::isinstance<py::list>(src)) {
				dumpList<py::list>(src, dst);
				return;
			}
			
			if(py::isinstance<py::dict>(src)) {
				auto asDict = py::cast<py::dict>(src);
				
				dst.beginObject(asDict.size());
				for(std::pair<py::handle, py::handle> entry : asDict) {
					dumpToVisitor(py::reinterpret_borrow<py::object>(entry.first), dst);
					dumpToVisitor(py::reinterpret_borrow<py::object>(entry.second), dst);
				}
				dst.endObject();
				return;
			}
			
			// Dynamic value
			DynamicValueReader r = py::cast<DynamicValueReader>(src);
			textio::save(r, dst);
		}
	}
	
	py::object Format::dumps(py::object r, bool compact, bool asBytes) {
		// Prepare write options
		textio::SaveOptions wopts;
		wopts.compact = compact;
		
		kj::VectorOutputStream os;
		auto v = this -> createVisitor(os, wopts);
		dumpToVisitor(r, *v);
		
		auto arr = os.getArray();
		
		if(asBytes) {
			return py::bytes((const char*) arr.begin(), arr.size());
		} else {
			return py::str((const char*) arr.begin(), arr.size());
		}
	}
	
	void Format::dump(py::object r, py::object o, bool compact) {
		int fd = py::cast<int>(o.attr("fileno")());
		kj::FdOutputStream os(fd);
		kj::BufferedOutputStreamWrapper buffered(os);
		
		// Flush python-side buffers before writing
		o.attr("flush")();
		
		// Prepare write options
		textio::SaveOptions wopts;
		wopts.compact = compact;
		
		auto v = this -> createVisitor(buffered, wopts);
		dumpToVisitor(r, *v);
	}
	
	TextIOFormat::TextIOFormat(const textio::Dialect& d) :
		dialect(d)
	{}
	
	Own<textio::Visitor> TextIOFormat::createVisitor(kj::BufferedOutputStream& os, const textio::SaveOptions& options) {
		return textio::createVisitor(os, dialect, options);
	}
	
	void TextIOFormat::read(textio::Visitor& dst, kj::BufferedInputStream& bis) {
		textio::load(bis, dst, dialect);
	}
	
	void FormattedReader::assign(const BuilderSlot& dst) {
		KJ_REQUIRE(!used, "Can only assign from a formatted load object once");
		used = true;
		
		auto t = dst.type;
		KJ_REQUIRE(t.isList() || t.isStruct(), "Can only assign to list and struct types");
		
		if(t.isList()) {
			auto initializer = [&](size_t s) {
				return dst.init(s).as<capnp::DynamicList>();
			};
			auto v = textio::createVisitor(t.asList(), initializer);
			parent.read(*v, *src);
		} else {
			auto v = textio::createVisitor(dst.init().as<capnp::DynamicStruct>());
			parent.read(*v, *src);
		}
	}
	
	py::object FormattedReader::read(py::object input) {
		PythonVisitor v(input);
		
		parent.read(v, *src);
		KJ_REQUIRE(v.done(), "Target was filled incompletely");
		
		return v.state().get<PythonVisitor::Done>().result;
	}
}

void initFormats(py::module_& m) {
	auto formatsMod = m.def_submodule("formats");
	
	py::class_<formats::FormattedReader, Assignable>(m,
		"FormattedReader",
		"A format-aware data stream. The object can be assigned to a struct"
		" / list builder or used as an argument to"
		" StructType.newMessage(...)."
	);
	
	py::class_<formats::Format>(m, "Format")		
		.def(&formats::Format::open, "open")
		
		.def(&formats::Format::dump, "dump")
		.def(&formats::Format::dumps, "dumps")
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