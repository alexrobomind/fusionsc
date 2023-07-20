namespace pybind11 { namespace detail {

template<>
struct type_caster<kj::Promise<py::object>> {
	PYBIND11_TYPE_CASTER(kj::Promise<py::object>, const_name("fsc.native.asnc.Future"));
	
	type_caster() :
		value(kj::NEVER_DONE)
	{}
	
	bool load(handle src, bool convert) {
		// Can only load from future-like objects
		if(!hasattr(src, "_asyncio_future_blocking")) {
			if(!hasattr(src, "__await__"))
				return false;
			
			// Python awaitables can be wrapped in tasks
			src = py::module_::import("asyncio").attr("create_task")(src).inc_ref();
		}
		
		value = fscpy::adaptAsyncioFuture(reinterpret_borrow<py::object>(src));
		return true;
	}
	
	static handle cast(kj::Promise<py::object> src, return_value_policy policy, handle parent) {
		return fscpy::convertToAsyncioFuture(kj::mv(src)).inc_ref();
	}
};

template<typename T>
struct type_caster<kj::Promise<T>> {
	using ValueConv = make_caster<T>;
	PYBIND11_TYPE_CASTER(kj::Promise<T>, const_name("fsc.native.asnc.Future[") + ValueConv::name + const_name("]"));
	
	using GenericPromise = kj::Promise<py::object>;
	
	type_caster() :
		value(kj::NEVER_DONE)
	{}
	
	bool load(handle src, bool convert) {
		type_caster<GenericPromise> baseCaster;
		if(!baseCaster.load(src, convert))
			return false;
		
		value = static_cast<GenericPromise&>(baseCaster)
		.then([](py::object o) {
			return py::cast<T>(kj::mv(o));
		});
		
		return true;
	}
		
	static handle cast(kj::Promise<T> src, return_value_policy policy, handle parent) {
		GenericPromise generic = src.then([](T&& o) {
			return py::cast(kj::mv(o));
		});
		return type_caster<GenericPromise>::cast(kj::mv(generic), policy, parent);
	}	
};

template<>
struct type_caster<kj::Promise<void>> {
	PYBIND11_TYPE_CASTER(kj::Promise<void>, const_name("fsc.native.asnc.Promise[NoneType]"));
	
	using GenericPromise = kj::Promise<py::object>;
	
	type_caster() :
		value(kj::NEVER_DONE)
	{}
	
	bool load(handle src, bool convert) {
		type_caster<GenericPromise> baseCaster;
		if(!baseCaster.load(src, convert))
			return false;
		
		value = static_cast<GenericPromise&>(baseCaster)
			.ignoreResult()
		;
		
		return true;
	}
		
	static handle cast(kj::Promise<void> src, return_value_policy policy, handle parent) {
		GenericPromise generic = src.then([]() -> py::object {
			return py::none();
		});
		return type_caster<GenericPromise>::cast(kj::mv(generic), policy, parent);
	}	
};

}}

namespace fscpy {

template<typename T>
T PythonWaitScope::wait(Promise<T>&& promise) {
	KJ_REQUIRE(canWait(), "Can not wait inside promises inside continuations or coroutines, and not in threads where no event loop was started.");
	
	auto restoreTo = activeScope;
	activeScope = nullptr;
	KJ_DEFER({activeScope = restoreTo;});
	
	return promise.wait(restoreTo -> waitScope);
}

}