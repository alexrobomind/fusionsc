#include "fscpy.h"

#include <kj/common.h>

namespace fscpy {
	namespace {
		void logInfo(bool enable) {
			if(enable)
				kj::_::Debug::setLogLevel(kj::LogSeverity::INFO);
			else
				kj::_::Debug::setLogLevel(kj::LogSeverity::WARNING);
		}
	}

	void raiseInPython(const kj::Exception& e) {	
		#define HANDLE_CASE(x) case kj::Exception::Type::x
		
		switch(e.getType()) {
			HANDLE_CASE(FAILED):
				PyErr_SetString(PyExc_RuntimeError, kj::str(e).cStr());
				return;
			
			HANDLE_CASE(OVERLOADED):
				excOverloaded(e.getDescription().cStr());
				return;
			
			HANDLE_CASE(DISCONNECTED):
				excDisconnected(e.getDescription().cStr());
				return;
			
			HANDLE_CASE(UNIMPLEMENTED):
				excUnimplemented(e.getDescription().cStr());
				return;
		}
		
		#undef HANDLE_CASE
		
		KJ_UNREACHABLE
	}

	kj::Exception convertPyError(py::error_already_set& e) {	
		auto formatException = py::module_::import("traceback").attr("format_exception");
		try {
			py::object t = e.type();
			py::object v = e.value();
			py::object tr = e.trace();
			
			kj::Exception::Type cppType = kj::Exception::Type::FAILED;
			
			if(t.is(excOverloaded)) {
				cppType = kj::Exception::Type::OVERLOADED;
			} else if(t.is(excDisconnected)) {
				cppType = kj::Exception::Type::DISCONNECTED;
			} else if(t.is(excUnimplemented)) {
				cppType = kj::Exception::Type::UNIMPLEMENTED;
			}
			
			auto pythonException = kj::strTree();
			
			switch(cppType) {
				case kj::Exception::Type::FAILED: {
					py::list formatted;
					if(t && tr)
						formatted = formatException(t, v, tr);
					else
						formatted = formatException(v);
					
					for(auto s : formatted) {
						pythonException = kj::strTree(mv(pythonException), py::cast<kj::StringPtr>(s), "\n");
					}
					
					break;
				}
				
				default:
					pythonException = kj::strTree(e.what());
			}
			
			return kj::Exception(::kj::Exception::Type::FAILED, __FILE__, __LINE__, pythonException.flatten());
		} catch(std::exception e2) {
			py::print("Failed to format exception", e.type(), e.value());
			auto exc = kj::getCaughtExceptionAsKj();
			return KJ_EXCEPTION(FAILED, "An underlying python exception could not be formatted due to the following error", exc);
		}
	}
	
	py::exception<kj::Exception> excOverloaded;
	py::exception<kj::Exception> excDisconnected;
	py::exception<kj::Exception> excUnimplemented;

	void initKj(py::module_& m) {
		py::module_ mkj = m.def_submodule("kj", "Python bindings for Cap'n'proto's 'kj' utility library");
		
		excOverloaded = py::exception<kj::Exception>(m, "OverloadError");
		excDisconnected = py::exception<kj::Exception>(m, "DisconnectError");
		excUnimplemented = py::exception<kj::Exception>(m, "Unimplemented");
		
		py::class_<kj::StringPtr>(mkj, "StringPtr", "C++ string container. Generally not returned, but can be subclassed")
			.def("__str__", [](kj::StringPtr ptr) { return ptr.cStr(); })
			.def("__repr__", [](kj::StringPtr ptr) { return ptr.cStr(); })
			.def("__eq__", [](kj::StringPtr self, kj::StringPtr other) { return self == other; }, py::is_operator())
		;
		py::class_<DynamicConstArray>(mkj, "ConstArray", "Immutable array")
			.def("__len__", &DynamicConstArray::size)
			.def("__getitem__", &DynamicConstArray::get)
			.def(
				"__iter__",
				[](DynamicConstArray& o) { return py::make_iterator(o.begin(), o.end()); },
				py::keep_alive<0, 1>()
			)
		;
		
		py::class_<DynamicArray, DynamicConstArray>(mkj, "Array", "Mutable array")
			.def("__setitem__", &DynamicArray::set)
		;
	
		// Translator for KJ exceptions
		py::register_exception_translator([](std::exception_ptr p) {
			try {
				if (p) std::rethrow_exception(p);
			} catch (kj::Exception& e) {
				auto description = kj::str(
					"C++ exception (", e.getType(), ") at ", e.getFile(), " -- line ", e.getLine(), "\n",
					e.getDescription(), "\n",
					"Trace: \n",
					kj::stringifyStackTrace(e.getStackTrace())
				);
				PyErr_SetString(PyExc_RuntimeError, description.cStr());
			}
		});
		
		mkj.def("logInfo", &logInfo);
	}
	
	DynamicConstArray::~DynamicConstArray() {}
}