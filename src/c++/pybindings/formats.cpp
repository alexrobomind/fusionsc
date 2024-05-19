#include "fscpy.h"
#include "assign.h"
#include "capnp.h"
#include "tensor.h"
#include "async.h"

#include "formats.h"

#include <pybind11/numpy.h>


namespace fscpy {

namespace {
	
	//! State struct for python visitor that indicates we are trying to fill a numpy array holder
	struct NewArray {
		using Entry = kj::OneOf<uint64_t, int64_t, double, py::object>;
		kj::Vector<Entry> entries;
		
		formats::ArrayHolder& holder;
		
		NewArray(formats::ArrayHolder& h) : holder(h) {}
		
		template<typename T>
		void finishAs() {
			py::array_t<T> result(std::array<size_t, 1>({entries.size()}));
			auto uc = result.mutable_unchecked();
			
			for(auto i : kj::indices(entries)) {
				T result;
				auto& e = entries[i];
				
				if(e.is<uint64_t>()) {
					result = (T) e.get<uint64_t>();
				} else if(e.is<int64_t>()) {
					result = (T) e.get<int64_t>();
				} else if(e.is<double>()) {
					result = (T) e.get<double>();
				} else {
					KJ_FAIL_REQUIRE("Internal error: Object in numeric array");
				}
				
				uc(i) = result;
			}
			
			holder.value = mv(result);
		}
		
		void finishAsObject() {
			// py::array_t<PyObject*> result(std::array<size_t, 1>({entries.size()}));
			py::list list;
						
			for(auto i : kj::indices(entries)) {
				PyObject* result;
				auto& e = entries[i];
				
				if(e.is<uint64_t>()) {
					result = py::cast(e.get<uint64_t>()).inc_ref().ptr();
				} else if(e.is<int64_t>()) {
					result = py::cast(e.get<int64_t>()).inc_ref().ptr();
				} else if(e.is<double>()) {
					result = py::cast(e.get<double>()).inc_ref().ptr();
				} else {
					result = e.get<py::object>().inc_ref().ptr();
				}
				
				list.append(result);
			}
			
			holder.value = py::array::ensure(list);
		}
		
		void finish() {
			// Determine datatype to use
			Entry::Tag tag = Entry::tagFor<uint64_t>();
			for(auto& e : entries)
				tag = std::max(tag, e.which());
			
			// KJ_DBG((unsigned int) tag);
			
			switch(tag) {
				case Entry::tagFor<uint64_t>(): finishAs<uint64_t>(); break;
				case Entry::tagFor<int64_t>(): finishAs<int64_t>(); break;
				case Entry::tagFor<double>(): finishAs<double>(); break;
				case Entry::tagFor<py::object>(): finishAsObject(); break;
				default: KJ_FAIL_REQUIRE("Internal error: Invalid tag for entry encountered");
			}
		}
	};
		
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
		
		using State = kj::OneOf<Preset, PresetList, NewList, Dict, Forward, NewArray, Done>;
		kj::Vector<State> states;
		
		State& state() { return states.back(); }
		
		bool done() override {
			return state().is<Done>();
		}
		
		void pop() {
			if(state().is<NewArray>()) {
				state().get<NewArray>().finish();
			}
			
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
				} else if(s.is<NewArray>()) {
					s = Done {py::cast(s.get<NewArray>().holder, py::return_value_policy::take_ownership)};
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
				py::object unwrapped = o;
				while(py::hasattr(unwrapped, "_fusionsc_wraps"))
					unwrapped = unwrapped.attr("_fusionsc_wraps");
				
				if(py::isinstance<DynamicStructBuilder>(unwrapped)) {
					states.add(Forward {textio::createVisitor(py::cast<DynamicStructBuilder>(unwrapped)), o});
					state().get<Forward>().visitor -> beginObject(s);
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
			} else if(state().is<NewArray>()) {
				py::dict newDict;
				state().get<NewArray>().entries.add((py::object&) newDict);
				states.add(Dict { newDict });
			}
		}
		
		void endObject() override {
			ACCEPT_FWD(endObject())
			pop();
		}
		
		void beginArray(Maybe<size_t> s) override {
			ACCEPT_FWD(beginArray(s));
			
			auto checkList = [&](py::object o) {
				// Check whether it is an array holder
				{
					py::detail::make_caster<formats::ArrayHolder> caster;
					if(caster.load(o, false)) {
						states.add(NewArray(caster));
						
						KJ_IF_MAYBE(pSize, s) {
							state().get<NewArray>().entries.reserve(*pSize);
						}
						return;
					}
				}
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
			} else if(state().is<NewArray>()) {
				py::list newList;
				state().get<NewArray>().entries.add((py::object&) newList);
				states.add(newList);
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
		void acceptNumber(T t) {
			if(state().is<NewArray>()) {
				state().get<NewArray>().entries.add(t);
				return;
			}
			
			acceptPrimitive(py::cast(t));
		}
		
		void acceptPrimitive(py::object asPy) {
			KJ_REQUIRE(!state().is<Done>());
			KJ_REQUIRE(!state().is<Forward>());
			
			if(state().is<Preset>()) {
				auto& ps = state().get<Preset>();
				KJ_REQUIRE(ps.object.is_none(), "Primitive value can only be unified with None");
				
				state() = Done { asPy };
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
			} else if(state().is<NewArray>()) {
				state().get<NewArray>().entries.add(asPy);
			}
		}
				
		void acceptInt(int64_t i) override {
			ACCEPT_FWD(acceptInt(i));
			acceptNumber(i);
		}
		
		void acceptUInt(uint64_t i) override {
			ACCEPT_FWD(acceptUInt(i));
			acceptNumber(i);
		}
		
		void acceptDouble(double d) override {
			ACCEPT_FWD(acceptDouble(d));
			acceptNumber(d);
		}
		
		void acceptString(kj::StringPtr s) override {
			ACCEPT_FWD(acceptString(s));
		
			// There are cases where strings can get converted into compound objects
			// To handle these correctly, we need to check whether the target is a struct
			// builder
			auto checkStruct = [&](py::object o) {
				while(py::hasattr(o, "_fusionsc_wraps"))
					o = o.attr("_fusionsc_wraps");
				
				if(py::isinstance<DynamicStructBuilder>(o)) {
					auto v = textio::createVisitor(py::cast<DynamicStructBuilder>(o));
					v -> acceptString(s);
					return true;
				}
				
				return false;
			};
			
			if(state().is<Preset>()) {
				auto& ps = state().get<Preset>();
				if(checkStruct(ps.object))
					return;
			} else if(state().is<PresetList>()) {
				auto& p = state().get<PresetList>();
				KJ_REQUIRE(p.offset < p.list.size(), "List too small");
				
				if(checkStruct(p.list[p.offset])) {
					++p.offset;
					return;
				}
			} else if(state().is<Dict>()) {
				auto& d = state().get<Dict>();
				KJ_IF_MAYBE(pKey, d.key) {
					if(d.dict.contains(*pKey) && checkStruct(d.dict[*pKey])) {
						d.key = nullptr;
						return;
					}
				}
			}
			
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
			py::object unwrapped = o;
			while(py::hasattr(unwrapped, "_fusionsc_wraps"))
				unwrapped = unwrapped.attr("_fusionsc_wraps");
			
			if(py::isinstance<DynamicStructBuilder>(unwrapped)) {
				auto& ds = py::cast<DynamicStructBuilder&>(unwrapped);
				states.add(Forward {textio::createVisitor(ds), o});
				return;
			}
			
			states.add(Preset {o});
		}

	};
	
	py::object readStream(kj::BufferedInputStream& is, py::object dst, textio::Dialect::Language lang) {		
		PythonVisitor v(dst);
		
		textio::load(is, v, lang);
		
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
		void dumpToVisitor(py::handle src, textio::Visitor& dst, const textio::SaveOptions& opts, Maybe<kj::WaitScope&> ws);
		
		void dumpNumpyArray(py::handle o, textio::Visitor& dst, const textio::SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
			py::list shape = o.attr("shape");
			py::object flat = o.attr("flatten")();
			dst.beginObject(2);
			
			dst.acceptString("shape");
			
			dst.beginArray(shape.size());
			for(auto el : shape)
				dumpToVisitor(el, dst, opts, ws);
			dst.endArray();
			
			dst.acceptString("data");
			
			dst.beginArray(py::len(flat));
			for(auto el : flat)
				dumpToVisitor(el, dst, opts, ws);
			dst.endArray();
			dst.endObject();
		}
		
		template<typename T>
		void dumpList(py::handle src, textio::Visitor& dst, const textio::SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
			auto asList = py::cast<T>(src);
			dst.beginArray(asList.size());
			for(auto e : asList)
				dumpToVisitor(py::reinterpret_borrow<py::object>(e), dst, opts, ws);
			dst.endArray();
		}
		
		void dumpToVisitor(py::handle src, textio::Visitor& dst, const textio::SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
			if(PyArray_Check(src.ptr())) {
				dumpNumpyArray(src, dst, opts, ws);
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
				dumpList<py::tuple>(src, dst, opts, ws);
				return;
			}
			
			if(py::isinstance<py::list>(src)) {
				dumpList<py::list>(src, dst, opts, ws);
				return;
			}
			
			if(py::isinstance<py::dict>(src)) {
				auto asDict = py::cast<py::dict>(src);
				
				dst.beginObject(asDict.size());
				for(std::pair<py::handle, py::handle> entry : asDict) {
					dumpToVisitor(py::reinterpret_borrow<py::object>(entry.first), dst, opts, ws);
					dumpToVisitor(py::reinterpret_borrow<py::object>(entry.second), dst, opts, ws);
				}
				dst.endObject();
				return;
			}
			
			// Wrapper types
			if(py::hasattr(src, "_fusionsc_wraps")) {
				dumpToVisitor(src.attr("_fusionsc_wraps"), dst, opts, ws);
				return;
			}
			
			// Dynamic value
			DynamicValueReader r = py::cast<DynamicValueReader>(src);
			textio::save(r, dst, opts, ws);
		}
		
		void dumpToStream(py::handle src, kj::BufferedOutputStream& os, Language lang, bool compact, Maybe<kj::WaitScope&> ws) {
			textio::SaveOptions opts;
			opts.compact = compact;
			
			auto v = textio::createVisitor(os, lang);
			dumpToVisitor(src, *v, opts, ws);
		}
	}
	
	py::object dumpToBytes(py::object o, Language lang, bool compact) {
		kj::VectorOutputStream os;
		dumpToStream(o, os, lang, compact, nullptr);
		
		auto arr = os.getArray();
		return py::bytes((const char*) arr.begin(), arr.size());
	}
	
	void dumpToFd(py::object o, int fd, Language lang, bool compact) {
		kj::FdOutputStream os(fd);
		kj::BufferedOutputStreamWrapper buffered(os);
		
		dumpToStream(o, buffered, lang, compact, nullptr);
		buffered.flush();
	}
	
	Promise<py::object> dumpAllToBytes(py::object o, Language lang, bool compact) {
		auto inner = [=](kj::WaitScope& ws) {
			kj::VectorOutputStream os;
			dumpToStream(o, os, lang, compact, ws);
			
			auto arr = os.getArray();
			return (py::object) py::bytes((const char*) arr.begin(), arr.size());
		};
		
		return kj::startFiber(1024 * 1024 * 8, mv(inner));
	}
	
	Promise<void> dumpAllToFd(py::object o, int fd, Language lang, bool compact)  {
		auto inner = [=](kj::WaitScope& ws) {
			kj::FdOutputStream os(fd);
			kj::BufferedOutputStreamWrapper buffered(os);
			
			dumpToStream(o, buffered, lang, compact, ws);
			buffered.flush();
		};
		
		return kj::startFiber(1024 * 1024 * 8, mv(inner));
	}
}

void initFormats(py::module_& m) {
	auto formatsMod = m.def_submodule("formats");
	
	py::class_<formats::ArrayHolder>(formatsMod, "ArrayHolder")
		.def(py::init<>())
		.def_readwrite("value", &formats::ArrayHolder::value)
	;
	
	formatsMod.def("readFd", &formats::readFd);
	formatsMod.def("readBuffer", &formats::readBuffer);
	
	formatsMod.def("dumpToBytes", &formats::dumpToBytes);
	formatsMod.def("dumpToFd", &formats::dumpToFd);
	
	formatsMod.def("dumpAllToBytes", &formats::dumpAllToBytes);
	formatsMod.def("dumpAllToFd", &formats::dumpAllToFd);
	
	py::enum_<formats::Language>(formatsMod, "Language")
		.value("YAML", formats::Language::YAML)
		.value("JSON", formats::Language::JSON)
		.value("CBOR", formats::Language::CBOR)
		.value("BSON", formats::Language::BSON)
		.value("MSGPACK", formats::Language::MSGPACK)
		.value("UBJSON", formats::Language::UBJSON)
	;
}

}