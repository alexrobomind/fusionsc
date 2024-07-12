#include "fscpy.h"
#include "assign.h"
#include "capnp.h"
#include "tensor.h"
#include "async.h"

#include "structio.h"

#include <pybind11/numpy.h>

#ifdef WIN32
#include <windows.h>
#include <fcntl.h>
#endif


namespace fscpy {

namespace {
	
	//! State struct for python visitor that indicates we are trying to fill a numpy array holder
	struct NewArray {
		using Entry = kj::OneOf<int64_t, uint64_t, double, py::object>;
		kj::Vector<Entry> entries;
		
		template<typename T>
		py::object finishAs() {
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
			
			return result;
		}
		
		py::object finishAsObject() {
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
			
			// return py::array::ensure(mv(list));
			return list;
		}
		
		py::object finish() {
			// Determine datatype to use
			Entry::Tag tag = Entry::tagFor<int64_t>();
			
			bool belowZero = false;
			for(auto& e : entries) {
				tag = std::max(tag, e.which());
				
				if(e.is<int64_t>()) {
					if(e.get<int64_t>() < 0) {
						belowZero = true;
					}
				}
			}
			
			// If we need uint for large integers and also have negative numbers,
			// we need to encode individually and fall back to Python's bigint en-
			// coding
			if(tag == Entry::tagFor<uint64_t>() && belowZero)
				tag = Entry::tagFor<py::object>();
						
			switch(tag) {
				case Entry::tagFor<uint64_t>(): return finishAs<uint64_t>();
				case Entry::tagFor<int64_t>(): return finishAs<int64_t>();
				case Entry::tagFor<double>(): return finishAs<double>();
				case Entry::tagFor<py::object>(): return finishAsObject();
				default: KJ_FAIL_REQUIRE("Internal error: Invalid tag for entry encountered");
			}
		}
	};
		
	struct PythonVisitor : public ::fsc::structio::Visitor {
		struct Preset { py::object object; };
		
		// using NewList = py::list;
		struct PresetList {
			py::list list;
			size_t offset = 0;
		};
		struct Dict {
			py::dict dict = py::dict();
			kj::Maybe<py::object> key = nullptr;
		};
		struct Forward {
			Own<::fsc::structio::Visitor> visitor;
			py::object original;
		};
		struct Done { py::object result; };
		
		Forward forwardToVoid() {
			return Forward {::fsc::structio::createVoidVisitor(), py::make_tuple()};
		}
		
		using State = kj::OneOf<Preset, PresetList, Dict, Forward, NewArray, Done>;
		kj::Vector<State> states;
		
		State& state() { return states.back(); }
		
		bool done() override {
			return state().is<Done>();
		}
		
		void pop() {			
			py::object result;
			
			{
				auto& s = state();
				switch(s.which()) {
					case State::tagFor<Preset>(): KJ_FAIL_REQUIRE("Error: pop() called on state Preset");
					case State::tagFor<PresetList>(): result = mv(s.get<PresetList>().list); break;
					case State::tagFor<Dict>(): result = mv(s.get<Dict>().dict); break;
					case State::tagFor<Forward>(): result = mv(s.get<Forward>().original); break;
					case State::tagFor<NewArray>(): result = s.get<NewArray>().finish(); break;
					case State::tagFor<Done>(): KJ_FAIL_REQUIRE("Error: pop() called on state Done");
				}
			}
						
			states.removeLast();
			
			// Check if object is a candidate for shaped array conversion
			if(py::isinstance<py::dict>(result)) {
				py::dict asDict = result;
				
				if(asDict.size() == 2 && asDict.contains("shape") && asDict.contains("data")) {
					try {
						py::object shape = asDict["shape"];
						py::array data = py::array::ensure(asDict["data"]);
						
						result = data.attr("reshape")(shape);
					} catch(py::cast_error& e) {
					} catch(py::error_already_set& e) {
					}
				}
			}
			
			if(states.size() > 0) {
				auto& s = state();
				
				switch(s.which()) {
					case State::tagFor<Preset>(): KJ_FAIL_REQUIRE("Internal error: pop() called inside Preset state");
					case State::tagFor<PresetList>(): {
						auto& asPreset = state().get<PresetList>();
						asPreset.list[asPreset.offset++] = mv(result);
						break;
					}
					case State::tagFor<Dict>(): {
						auto& dict = state().get<Dict>();
						KJ_IF_MAYBE(pKey, dict.key) {
							py::object key = mv(*pKey);
							dict.dict[key] = mv(result);
							dict.key = nullptr;
						} else {
							KJ_FAIL_REQUIRE("Internal error: pop() called in dict key without key set");
						}
						break;
					}
					case State::tagFor<Forward>(): KJ_FAIL_REQUIRE("Internal error: pop() called inside Forward state");
					case State::tagFor<NewArray>(): {
						auto& asNA = state().get<NewArray>();
						asNA.entries.add(mv(result));
						break;
					}
					case State::tagFor<Done>(): KJ_FAIL_REQUIRE("Error: pop() called on done visitor");
				}
			} else {
				states.add(Done { mv(result) });
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
				if(o.is_none()) {
					states.add(Dict {py::dict()});
					return;
				}
				
				if(py::isinstance<py::tuple>(o) && py::len(o) == 0) {
					states.add(forwardToVoid());
					state().get<Forward>().visitor -> beginObject(s);
					return;
				}
				
				py::object unwrapped = o;
				while(py::hasattr(unwrapped, "_fusionsc_wraps"))
					unwrapped = unwrapped.attr("_fusionsc_wraps");
				
				if(py::isinstance<DynamicStructBuilder>(unwrapped)) {
					states.add(Forward {::fsc::structio::createVisitor(py::cast<DynamicStructBuilder>(unwrapped)), o});
					state().get<Forward>().visitor -> beginObject(s);
				} else if(py::isinstance<py::dict>(o)) {
					states.add(Dict {o});
				} else {
					std::string typeName = py::str(py::type::of(o));
					KJ_FAIL_REQUIRE("Target objects for mappings must be None, structs (incl. wrappers) or dicts", typeName);
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
			} else if(state().is<PresetList>()) {
				auto& p = state().get<PresetList>();
				KJ_REQUIRE(p.list.size() > p.offset, "List to small to add to");
				
				py::object entry = p.list[p.offset];
				if(entry.is_none()) {
					states.add(Dict { py::dict() });
				} else {
					checkObject(entry);
				}
			} else if(state().is<Dict>()) {
				auto& dict = state().get<Dict>();
				KJ_IF_MAYBE(pKey, dict.key) {
					py::object& key = *pKey;
										
					if(dict.dict.contains(key)) {
						checkObject(dict.dict[key]);
					} else {
						states.add(Dict {py::dict()});
					}
				} else {
					KJ_FAIL_REQUIRE("Map key must be int, float, or str, not map");
				}
			} else if(state().is<NewArray>()) {
				states.add(Dict { py::dict() });
			}
		}
		
		void endObject() override {
			ACCEPT_FWD(endObject())
			pop();
		}
		
		void beginArray(Maybe<size_t> s) override {
			ACCEPT_FWD(beginArray(s));
			
			auto checkList = [&](py::object o) {
				if(o.is_none()) {
					states.add(NewArray());
				} else if(py::isinstance<py::list>(o)) {
					states.add(PresetList {o});
				} else if(py::isinstance<py::tuple>(o) && py::len(o) == 0) {
					states.add(forwardToVoid());
					state().get<Forward>().visitor -> beginArray(s);
				} else {
					std::string typeName = py::str(py::type::of(o));
					KJ_FAIL_REQUIRE("Targets for array objects must be None or list", typeName);
				}
			};
			
			if(state().is<Preset>()) {
				py::object o = state().get<Preset>().object;
				if(o.is_none()) {
					state().init<NewArray>();
				} else {
					states.removeLast();
					checkList(o);
				}
			} else if(state().is<PresetList>()) {
				auto& p = state().get<PresetList>();
				KJ_REQUIRE(p.list.size() > p.offset, "List to small to add to");
				
				py::object entry = p.list[p.offset];
				if(entry.is_none()) {
					states.add(NewArray());
				} else {
					checkList(entry);
				}
			} else if(state().is<Dict>()) {
				auto& dict = state().get<Dict>();
				KJ_IF_MAYBE(pKey, dict.key) {
					py::object& key = *pKey;
										
					if(dict.dict.contains(key)) {
						checkList(dict.dict[key]);
					} else {
						states.add(NewArray());
					}
				} else {
					KJ_FAIL_REQUIRE("Map key must be int, float, or str, not list");
				}
			} else if(state().is<NewArray>()) {
				states.add(NewArray());
			}
			
			KJ_IF_MAYBE(pSize, s) {
				if(state().is<NewArray>()) {
					state().get<NewArray>().entries.reserve(*pSize);
				} else if(state().is<PresetList>()) {
					KJ_REQUIRE(*pSize == state().get<PresetList>().list.size(), "Mismatch between given list and input");
				} else {
					KJ_FAIL_REQUIRE("Internal error: state() must be PresetList or NewArray");
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
				
				if(py::isinstance<py::tuple>(ps.object) && py::len(ps.object) == 0) {
					state() = Done { ps.object };
				} else if(ps.object.is_none()) {
					state() = Done { asPy };
				} else {
					KJ_FAIL_REQUIRE("Primitive value can only be unified with None or ()");
				}
			} else if(state().is<PresetList>()) {
				auto& p = state().get<PresetList>();
				
				KJ_REQUIRE(p.offset < p.list.size(), "List too small");
				
				auto ps = p.list[p.offset];
				
				if(py::isinstance<py::tuple>(ps) && py::len(ps) == 0) {
					p.list[p.offset++] = ps;
				} else if(ps.is_none()) {
					p.list[p.offset++] = asPy;
				} else {
					KJ_FAIL_REQUIRE("Primitive value can only be unified with None or ()");
				}
			} else if(state().is<Dict>()) {
				auto& d = state().get<Dict>();
				KJ_IF_MAYBE(pKey, d.key) {					
					if(d.dict.contains(*pKey)) {				
						auto ps = d.dict[*pKey];
						
						if(py::isinstance<py::tuple>(ps) && py::len(ps) == 0) {
							d.dict[*pKey] = ps;
						} else if(ps.is_none()) {
							d.dict[*pKey] = asPy;
						} else {
							KJ_FAIL_REQUIRE("Primitive value can only be unified with None or ()");
						}
					} else {
						d.dict[*pKey] = asPy;
					}
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
			
			// Note: The right operand gets implicitly
			// converted to uint64_t for the comparison
			if(i <= (int64_t) kj::maxValue)
				acceptNumber((int64_t) i);
			else
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
					auto v = ::fsc::structio::createVisitor(py::cast<DynamicStructBuilder>(o));
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
				states.add(Forward {::fsc::structio::createVisitor(ds), o});
				return;
			}
			
			states.add(Preset {o});
		}

	};
	
	py::object readStream(kj::BufferedInputStream& is, py::object dst, ::fsc::structio::Dialect::Language lang) {		
		PythonVisitor v(dst);
		
		::fsc::structio::load(is, v, lang);
		
		KJ_REQUIRE(v.done(), "Target was filled incompletely");
		return v.state().get<PythonVisitor::Done>().result;
	}
}

namespace structio {
	#ifdef WIN32
	using StructioFd = intptr_t;
	
	HANDLE cloneHandle(HANDLE in) {
		HANDLE proc = GetCurrentProcess();
		HANDLE out;

		DuplicateHandle(
			proc, in, proc, &out, 
			0, FALSE, DUPLICATE_SAME_ACCESS
		);
		return out;
	}
	
	#else
	using StructioFd = int;
	#endif
	
	py::object readFd(StructioFd inputFd, py::object dst, Language lang) {
		#ifdef WIN32
			int fd = _open_osfhandle((intptr_t) cloneHandle(reinterpret_cast<HANDLE>(inputFd)), _O_RDONLY);
			KJ_REQUIRE(fd != -1, "Failed to open OS handle");
			KJ_DEFER({ _close(fd); });
		# else
			int fd = inputFd;
		#endif
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
		void dumpToVisitor(py::handle src, ::fsc::structio::Visitor& dst, const ::fsc::structio::SaveOptions& opts, Maybe<kj::WaitScope&> ws);
		
		void dumpNumpyArray(py::handle o, ::fsc::structio::Visitor& dst, const ::fsc::structio::SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
			py::list shape = o.attr("shape");
			py::object flat = o.attr("flatten")();
			
			// 1D-arrays are written as lists
			if(shape.size() == 1) {
				dst.beginArray(py::len(flat));
				for(auto el : flat)
					dumpToVisitor(el, dst, opts, ws);
				dst.endArray();
				return;
			}
			
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
		void dumpList(py::handle src, ::fsc::structio::Visitor& dst, const ::fsc::structio::SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
			auto asList = py::cast<T>(src);
			dst.beginArray(asList.size());
			for(auto e : asList)
				dumpToVisitor(py::reinterpret_borrow<py::object>(e), dst, opts, ws);
			dst.endArray();
		}
		
		void dumpToVisitor(py::handle src, ::fsc::structio::Visitor& dst, const ::fsc::structio::SaveOptions& opts, Maybe<kj::WaitScope&> ws) {
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
			::fsc::structio::save(r, dst, opts, ws);
		}
		
		void dumpToStream(py::handle src, kj::BufferedOutputStream& os, Language lang, bool compact, Maybe<kj::WaitScope&> ws) {
			::fsc::structio::SaveOptions opts;
			opts.compact = compact;
			
			auto v = ::fsc::structio::createVisitor(os, lang);
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

void initStructio(py::module_& m) {
	auto structioMod = m.def_submodule("structio");
	
	structioMod.def("readFd", &structio::readFd);
	structioMod.def("readBuffer", &structio::readBuffer);
	
	structioMod.def("dumpToBytes", &structio::dumpToBytes);
	structioMod.def("dumpToFd", &structio::dumpToFd);
	
	structioMod.def("dumpAllToBytes", &structio::dumpAllToBytes);
	structioMod.def("dumpAllToFd", &structio::dumpAllToFd);
	
	py::enum_<structio::Language>(structioMod, "Language")
		.value("YAML", structio::Language::YAML)
		.value("JSON", structio::Language::JSON)
		.value("CBOR", structio::Language::CBOR)
		.value("BSON", structio::Language::BSON)
		.value("MSGPACK", structio::Language::MSGPACK)
		.value("UBJSON", structio::Language::UBJSON)
	;
}

}