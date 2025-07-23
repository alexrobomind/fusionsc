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
		auto description =
				e.getDescription().startsWith("C++ exception (")
			?
				kj::str(e.getDescription())
			:
				kj::str(
					"C++ exception (", e.getType(), ") at ", e.getFile(), " -- line ", e.getLine(), "\n",
					e.getDescription(), "\n",
					"Trace: \n",
					kj::stringifyStackTrace(e.getStackTrace())
				)
			;
			
		
		#define HANDLE_CASE(x) case kj::Exception::Type::x
		
		switch(e.getType()) {
			HANDLE_CASE(FAILED):
				PyErr_SetString(PyExc_RuntimeError, description.cStr());
				return;
				
			// Note: Calling py::exception::operator() to set the python exception
			// is deprecated (and slightly unclear) but I will keep it in here for
			// now to keep backwards compatibility to older pybind11 versions.
			//
			// py::set_error(...) (the replacement) is rather new
			
			HANDLE_CASE(OVERLOADED):
				excOverloaded(description.cStr());
				return;
			
			HANDLE_CASE(DISCONNECTED):
				excDisconnected(description.cStr());
				return;
			
			HANDLE_CASE(UNIMPLEMENTED):
				excUnimplemented(description.cStr());
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
			
			if(e.matches(excOverloaded)) {
				cppType = kj::Exception::Type::OVERLOADED;
			} else if(e.matches(excDisconnected)) {
				cppType = kj::Exception::Type::DISCONNECTED;
			} else if(e.matches(excUnimplemented)) {
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
			
			return kj::Exception(/*::kj::Exception::Type::FAILED*/cppType, __FILE__, __LINE__, pythonException.flatten());
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
		
		excOverloaded = py::exception<kj::Exception>(mkj, "OverloadError", PyExc_RuntimeError);
		excDisconnected = py::exception<kj::Exception>(mkj, "DisconnectError", PyExc_RuntimeError);
		excUnimplemented = py::exception<kj::Exception>(mkj, "UnimplementedError", PyExc_RuntimeError);
		
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
				/*auto description = kj::str(
					"C++ exception (", e.getType(), ") at ", e.getFile(), " -- line ", e.getLine(), "\n",
					e.getDescription(), "\n",
					"Trace: \n",
					kj::stringifyStackTrace(e.getStackTrace())
				);
				PyErr_SetString(PyExc_RuntimeError, description.cStr());*/
				raiseInPython(e);
			}
		});
		
		mkj.def("logInfo", &logInfo);
	}
	
	DynamicConstArray::~DynamicConstArray() {}
}