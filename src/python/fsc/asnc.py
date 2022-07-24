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

def asyncFunction(f: Callable[P, Awaitable[T]]) -> Callable[P, Promise[T]]:
	"""
	Decorator. Transforms a function returning a coroutine into a function
	returning a promise.
	"""
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		return run(f(*args, **kwargs))
	
	return wrapper

def eager(f: Callable[P, Awaitable[T]]) -> Callable[P, T]:
	"""
	Transforms a function returning a coroutine or promise into
	one that immediately executes via the main event loop.
	"""
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		return wait(f(*args, **kwargs))
	
	return wrapper