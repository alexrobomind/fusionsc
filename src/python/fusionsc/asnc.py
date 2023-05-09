"""
Asynchronous processing (promises, coroutines)
"""

from .native.asnc import (
	Promise, # This class is untyped in the C++ library, when referring to it in type hints, use Strings
	
	startEventLoop,
	stopEventLoop,
	hasEventLoop,
	cycle,
	
	run,
	
	FiberPool
)

from .native.timer import delay

from typing import Callable, Any, Union, TypeVar, Awaitable
from typing_extensions import ParamSpec

import functools
import inspect

T = TypeVar("T")
P = ParamSpec("P")

__all__ = ["Promise", "startEventLop", "stopEventLoop", "hasEventLoop", "FiberPool", "wait", "asyncFunction", "run"]

class AsyncMethodDescriptor:
	"""Helper class to implement asyncFunction"""
	def __init__(self, f):
		self.f = f
	
	def __call__(self, *args, **kwargs):
		coro = self.f(*args, **kwargs)
		return run(coro).wait()
	
	def asnc(self, *args, **kwargs):
		coro = self.f(*args, **kwargs)
		return run(coro)
	
	def __get__(self, obj, objtype = None):
		return AsyncMethodDescriptor(self.f.__get__(obj, objtype))
	
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
				
		if (orig is Promise or orig is Awaitable) and len(args) == 1:
			return sig.replace(return_annotation = args[0])
		else:
			return sig.replace(return_annotation = Any)

def wait(awaitable: Awaitable[T]) -> T:
	"""
	Awaits a coroutine result by running the coroutine on the main event loop.
	"""
	return run(awaitable).wait()

def asyncFunction(f: Callable[P, Awaitable[T]]) -> Callable[P, T]:
	"""
	_`Asynchronous Function`
	
	  Wraps a coroutine or promise function as a usual eagerly evaluate function.
	  The asynchronous function can still be accessed using the *async* property.
	"""
	return functools.wraps(f)(AsyncMethodDescriptor(f))