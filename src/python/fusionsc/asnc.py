"""
Asynchronous processing (promises, coroutines)
"""

from .native.asnc import (
	Future, # This class is untyped in the C++ library, when referring to it in type hints, use Strings
	Executor,
	
	startEventLoop,
	stopEventLoop,
	hasEventLoop,
	cycle,
	
	canWait,
	
	FiberPool,
	
	eventLoopDict
)

from .native.asnc import wait as nativeWait

# Note: The timer functionality is not yet supported
#from .native.timer import delay

from typing import Callable, Any, Union, TypeVar, Awaitable
from typing_extensions import ParamSpec

import functools
import inspect
import asyncio
import threading

T = TypeVar("T")
P = ParamSpec("P")

__all__ = ["Future", "startEventLop", "stopEventLoop", "hasEventLoop", "FiberPool", "wait", "asyncFunction", "run"]

class AsyncMethodDescriptor:
	"""Helper class to implement asyncFunction"""
	def __init__(self, f):
		self.f = f
	
	def __call__(self, *args, **kwargs):
		assert hasEventLoop(), "Asynchronous objects can only be used in threads with an active event loop. Create one for your thread with fsc.asnc.startEventLoop()"
		assert canWait(), (
			"The synchronous call for this function is only valid in areas where the program can safely suspend execution. Particularly, coroutines "
			"decorated with fsc.asnc.asyncFunction or run by fsc.asnc.run(...) may not use the synchronous wrappers. Instead of calling f(...),  call "
			"the asynchronous variant f.asnc(...) and await the result. If you absolutely need to call the synchronous version (e.g. because you "
			"are inside a call by an external library that has doesn't accept awaitables), then use the class fsc.asnc.FiberPool to start a new fiber "
			"(which gets a new stack) which can suspend synchronous-style."
		)
		
		future = self.asnc(*args, **kwargs)
		return wait(future)
	
	def asnc(self, *args, **kwargs):
		assert hasEventLoop(), "Asynchronous objects can only be used in threads with an active event loop. Create one for your thread with fsc.asnc.startEventLoop()"
		return self.f(*args, **kwargs)
	
	def __get__(self, obj, objtype = None):
		return functools.wraps(self.f)(AsyncMethodDescriptor(self.f.__get__(obj, objtype)))
	
	@property
	def __doc__(self):
		docSuffix = "*Note* Has :ref:`asynchronous variant<Asynchronous Function>` '.asnc(...)' that returns Promise[...]"
		
		if(hasattr(self.f, "__doc__") and self.f.__doc__ is not None):
			return self.f.__doc__ + "\n" + docSuffix
		
		return docSuffix
	
	@__doc__.setter
	def __doc__(self, val):
		pass
	
	@property
	def __signature__(self):
		import typing
		import inspect
		
		sig = inspect.signature(self.f)
		retAnn = sig.return_annotation
		
		orig = typing.get_origin(retAnn)
		args = typing.get_args(retAnn)
				
		if (orig is Future or orig is Awaitable) and len(args) == 1:
			return sig.replace(return_annotation = args[0])
		else:
			return sig.replace(return_annotation = Any)

def wait(awaitable: Awaitable[T]) -> T:
	"""
	Awaits a coroutine result by running the coroutine on the main event loop.
	"""
	assert hasEventLoop(), "Asynchronous objects can only be used in threads with an active event loop. Create one for your thread with fsc.asnc.startEventLoop()"
	assert canWait(), (
		"Can not wait on promises / coroutine because the program can not safely suspend execution. Particularly, coroutines "
		"decorated with fsc.asnc.asyncFunction or run by fsc.asnc.run(...) may not use wait(...). If you absolutely need to call the synchronous version (e.g. because you "
		"are inside a call by an external library that has doesn't accept awaitables), then use the class fsc.asnc.FiberPool to start a new fiber "
		"(which gets a new stack) which can suspend synchronous-style."
	)
			
	return nativeWait(awaitable)

def asyncFunction(f: Callable[P, Awaitable[T]]) -> Callable[P, T]:
	"""
	_`Asynchronous Function`
	
	  Wraps a coroutine or promise function as a usual eagerly evaluate function.
	  The asynchronous function can still be accessed using the *async* property.
	"""
	return functools.wraps(f)(AsyncMethodDescriptor(f))

class EventLoopLocal:
	def __init__(self, default = None):
		self.default = default
	
	@property
	def value(self):
		return eventLoopDict().get(id(self), self.default)
	
	@value.setter
	def value(self, val):
		eventLoopDict()[id(self)] = val
	
	@value.deleter
	def value(self):
		del eventLoopDict()[id(self)]
	
	def isSet(self):
		return id(self) in eventLoopDict()