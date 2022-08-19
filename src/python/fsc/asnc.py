from .native.asnc import (
	Promise, # This class is untyped in the C++ library, when referring to it in type hints, use Strings
	
	startEventLoop,
    stopEventLoop,
	hasEventLoop,
	cycle,
	
	run
)

from typing import Callable, Any, Union, TypeVar, Awaitable
from typing_extensions import ParamSpec

import functools

T = TypeVar("T")
P = ParamSpec("P")

class AsyncMethodDescriptor:
	def __init__(self, f):
		self.f = f
	
	def __call__(self, *args, **kwargs):
		coro = self.f(*args, **kwargs)
		return run(coro).wait()
	
	def asnc(self, *args, **kwargs):
		coro = self.f(*args, **kwargs)
		return run(coro)
	
	def __get__(self, obj, objtype = None):
		if hasattr(self.f, "__get__"):
			f = self.f.__get__(obj, objtype)
		else:
			f = functools.partial(self.f, obj)		
		
		@functools.wraps(f)
		def wrapper(*args, **kwargs):
			coro = f(*args, **kwargs)
			return run(coro).wait()
		
		@functools.wraps(f)
		def asnc(*args, **kwargs):
			coro = f(*args, **kwargs)
			return run(coro)
		
		wrapper.asnc = asnc
		return wrapper
		

def wait(awaitable: Awaitable[T]) -> T:
	"""
	Awaits a coroutine result by running the coroutine on the main event loop.
	"""
	return run(awaitable).wait()

def asyncFunction(f: Callable[P, Awaitable[T]]) -> Callable[P, T]:
	return AsyncMethodDescriptor(f)