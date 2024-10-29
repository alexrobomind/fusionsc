namespace pybind11 { namespace detail {

template<>
struct type_caster<kj::Promise<py::object>> {
	PYBIND11_TYPE_CASTER(kj::Promise<py::object>, const_name("fsc.native.asnc.Future"));
	
	type_caster() :
		value(kj::NEVER_DONE)
	{}
	
	bool load(handle src, bool convert) {
		// Can only load from future-like objects or awaitables
		if(!hasattr(src, "_asyncio_future_blocking") && !hasattr(src, "__await__"))
			return false;
		
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
	PYBIND11_TYPE_CASTER(kj::Promise<void>, const_name("fsc.native.asnc.Future[NoneType]"));
	
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
	// TODO: Check if an event port exists (instead of requiring canWait).
	// TODO: If we have no WaitScope, delegate execution to the asyncio event loop and pray.
	
	KJ_REQUIRE(canWait(), "Can not wait inside promises inside continuations or coroutines, and not in threads where no event loop was started.");
	KJ_REQUIRE(PyGILState_Check(), "Can only wait inside GIL");
	
	auto restoreTo = activeScope;
	activeScope = nullptr;
	KJ_DEFER({activeScope = restoreTo;});
	
	auto eventLoop = py::module_::import("asyncio").attr("get_event_loop")();
	AsyncioEventPort::adjustEventLoop(eventLoop);
	
	while(!promise.poll(restoreTo -> waitScope)) {
		// While the python event loop runs, we can wait inside it again
		// This way, we can do reentrant waits on the Python side.
		//
		// Note: On the C++ side, we still need fibers though.
		activeScope = restoreTo;
		KJ_DEFER({activeScope = nullptr;});
		AsyncioEventPort::waitForEvents();
	}
	
	return promise.wait(restoreTo -> waitScope);
}

template<typename P>
bool PythonWaitScope::poll(P&& promise) {
	KJ_REQUIRE(canWait(), "Can not wait inside promises inside continuations or coroutines, and not in threads where no event loop was started.");
	KJ_REQUIRE(PyGILState_Check(), "Can only wait inside GIL");
	
	auto restoreTo = activeScope;
	activeScope = nullptr;
	KJ_DEFER({activeScope = restoreTo;});
	
	return promise.poll(restoreTo -> waitScope);
}

}
