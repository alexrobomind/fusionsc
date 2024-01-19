#include "fscpy.h"
#include "assign.h"
#include "capnp.h"
#include "tensor.h"

#include "formats.h"


namespace fscpy {

namespace {
	struct PythonVisitor : public textio::Visitor {
		struct Preset { py::object object; };
		
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
		
		using State = kj::OneOf<Preset, PresetList, NewList, Dict, Forward, Done>;
		kj::Vector<State> states;
		
		State& state() { return states.back(); }
		
		bool done() override {
			return state().is<Done>();
		}
		
		void pop() {
			if(states.size() > 1) {
				states.removeLast();
			} else {
				auto& s = state();
				
				KJ_REQUIRE(!s.is<Done>());
				
				if(s.is<Preset>()) {
					s = Done { s.get<Preset>().object };
				} else if(s.is<NewList>()) {
					s = Done { s.get<NewList>() };
				} else if(s.is<PresetList>()) {
					s = Done {s.get<PresetList>().list};
				} else if(s.is<Dict>()) {
					s = Done {s.get<Dict>().dict};
				} else if(s.is<Forward>()) {
					s = Done {s.get<Forward>().original};
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
			

			if(state().is<Preset>()) {
				py::object o = state().get<Preset>().object;
				
				if(o.is_none()) {
					state().init<Dict>();
				} else {
					states.removeLast();
					checkObject(o);
				}	
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
					
					p.list[p.offset++] = newDict;
				} else {
					++p.offset;
					checkObject(entry);
				}
			} else if(state().is<Dict>()) {
				auto& dict = state().get<Dict>();
				KJ_IF_MAYBE(pKey, dict.key) {
					py::object key = mv(*pKey);
					dict.key = nullptr;
					
					if(dict.dict.contains(key)) {
						checkObject(dict.dict[key]);
					} else {
						py::dict newDict;
						dict.dict[key] = newDict;
						states.add(Dict {newDict});
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
				states.add(PresetList {o});
			};
			
			if(state().is<Preset>()) {
				py::object o = state().get<Preset>().object;
				if(o.is_none()) {
					state().init<NewList>();
				} else {
					states.removeLast();
					checkList(o);
				}
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
					p.list[p.offset++] = newList;
					states.add(newList);
				} else {
					++p.offset;
					checkList(entry);
				}
			} else if(state().is<Dict>()) {
				auto& dict = state().get<Dict>();
				KJ_IF_MAYBE(pKey, dict.key) {
					py::object key = mv(*pKey);
					dict.key = nullptr;
					
					if(dict.dict.contains(key)) {
						checkList(dict.dict[key]);
					} else {
						py::list newList;
						dict.dict[key] = newList;
						states.add(newList);
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
		
		void acceptPrimitive(py::object asPy) {
			KJ_REQUIRE(!state().is<Done>());
			KJ_REQUIRE(!state().is<Forward>());
			
			if(state().is<Preset>()) {
				auto& ps = state().get<Preset>();
				KJ_REQUIRE(ps.object.is_none(), "Primitive value can only be unified with None");
				ps.object = asPy;
			} else if(state().is<PresetList>()) {
				auto& p = state().get<PresetList>();
				
				KJ_REQUIRE(p.offset < p.list.size(), "List too small");
				KJ_REQUIRE(p.list[p.offset].is_none(), "Primitive value can only be unified with None");
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
			acceptPrimitive(py::cast(i));
		}
		
		void acceptUInt(uint64_t i) override {
			ACCEPT_FWD(acceptUInt(i));
			acceptPrimitive(py::cast(i));
		}
		
		void acceptDouble(double d) override {
			ACCEPT_FWD(acceptDouble(d));
			acceptPrimitive(py::cast(d));
		}
		
		void acceptString(kj::StringPtr s) override {
			ACCEPT_FWD(acceptString(s));
			acceptPrimitive(py::cast(s));
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
			acceptPrimitive(py::cast(b));
		}
		
		PythonVisitor(py::object o) {
			if(py::isinstance<DynamicStructBuilder>(o)) {
				auto& ds = py::cast<DynamicStructBuilder&>(o);
				states.add(Forward {textio::createVisitor(ds), o});
				return;
			}
			
			states.add(Preset {o});
		}

	};
	
	py::object readStream(kj::BufferedInputStream& is, py::object dst, textio::Dialect::Language lang) {
		textio::Dialect dialect;
		dialect.language = lang;
		
		PythonVisitor v(dst);
		
		textio::load(is, v, dialect);
		
		KJ_REQUIRE(v.done(), "Target was filled incompletely");
		return v.state().get<PythonVisitor::Done>().result;
	}
}

namespace formats {		
	py::object readFd(int fd, py::object dst, Language lang) {
		kj::FdInputStream is(fd);
		kj::BufferedInputStreamWrapper buffered(is);
		
		return readStream(buffered, dst, lang);
	}
	
	py::object readBuffer(py::buffer buffer, py::object dst, Language lang) {
		py::buffer_info info = buffer.request();
		KJ_REQUIRE(info.itemsize == 1, "Can only read from character buffers");
		
		kj::ArrayInputStream is(ArrayPtr<const kj::byte>((const kj::byte*) info.ptr, info.size));
		return readStream(is, dst, lang);
	}
	
	namespace {
		void dumpToVisitor(py::handle src, textio::Visitor& dst);
		
		void dumpNumpyArray(py::handle o, textio::Visitor& dst) {
			py::list shape = o.attr("shape");
			py::object flat = o.attr("flatten")();
			dst.beginObject(2);
			
			dst.acceptString("shape");
			
			dst.beginArray(shape.size());
			for(auto el : shape)
				dumpToVisitor(el, dst);
			dst.endArray();
			
			dst.acceptString("data");
			
			dst.beginArray(py::len(flat));
			for(auto el : flat)
				dumpToVisitor(el, dst);
			dst.endArray();
			dst.endObject();
		}
		
		template<typename T>
		void dumpList(py::handle src, textio::Visitor& dst) {
			auto asList = py::cast<T>(src);
			dst.beginArray(asList.size());
			for(auto e : asList)
				dumpToVisitor(py::reinterpret_borrow<py::object>(e), dst);
			dst.endArray();
		}
		
		void dumpToVisitor(py::handle src, textio::Visitor& dst) {
			if(PyArray_Check(src.ptr())) {
				dumpNumpyArray(src, dst);
				return;
			}
			
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
		
		void dumpToStream(py::handle src, kj::BufferedOutputStream& os, Language lang, bool compact) {
			textio::SaveOptions opts;
			opts.compact = compact;
			
			textio::Dialect dialect;
			dialect.language = lang;
			
			auto v = textio::createVisitor(os, dialect, opts);
			dumpToVisitor(src, *v);
		}
	}
	
	py::object dumpToBytes(py::object o, Language lang, bool compact) {
		kj::VectorOutputStream os;
		dumpToStream(o, os, lang, compact);
		
		auto arr = os.getArray();
		return py::bytes((const char*) arr.begin(), arr.size());
	}
	
	void dumpToFd(py::object o, int fd, Language lang, bool compact) {
		kj::FdOutputStream os(fd);
		kj::BufferedOutputStreamWrapper buffered(os);
		
		dumpToStream(o, buffered, lang, compact);
		buffered.flush();
	}
}

void initFormats(py::module_& m) {
	auto formatsMod = m.def_submodule("formats");
	
	formatsMod.def("readFd", &formats::readFd);
	formatsMod.def("readBuffer", &formats::readBuffer);
	
	formatsMod.def("dumpToBytes", &formats::dumpToBytes);
	formatsMod.def("dumpToFd", &formats::dumpToFd);
	
	py::enum_<formats::Language>(formatsMod, "Language")
		.value("YAML", formats::Language::YAML)
		.value("JSON", formats::Language::JSON)
		.value("CBOR", formats::Language::CBOR)
		.value("BSON", formats::Language::BSON)
	;
}

}