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

def wait(awaitable: Awaitable[T]) -> T:
	"""
	Awaits a coroutine result by running the coroutine on the main event loop.
	"""
	return run(awaitable).wait()

def asyncFunction(f: Callable[P, Awaitable[T]]) -> Callable[P, T]:
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		coro = f(*args, **kwargs)
		return run(coro).wait()
	
	@functool.wraps(f)
	def asnc(*args, **kwargs):
		coro = f(*args, **kwargs)
		return run(coro)
	
	wrapper.asnc = asnc
	return wrapper