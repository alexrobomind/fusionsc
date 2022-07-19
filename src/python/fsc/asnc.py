from . import native

from typing import Coroutine, Callable, Any, Union

import functools

def run(coroutine: Union[native.Promise, Coroutine[native.Promise, Any, Any]]) -> native.Promise:
	"""
	Transforms a coroutine result (that can in turn await promises)
	into a promise
	"""
	return native.run(coroutine)

def wait(coroutine: Union[native.Promise, Coroutine[native.Promise, Any, Any]]) -> Any:
	"""
	Awaits a coroutine result by running the coroutine on the main event loop.
	"""
	return native.run(coroutine).wait()

def asyncFunction(f: Callable[..., Union[native.Promise, Coroutine[native.Promise, Any, Any]]]) -> Callable[..., native.Promise]:
	"""
	Transforms a function returning a coroutine into a function
	returning a promise.
	"""
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		return run(f(*args, **kwargs))
	
	return wrapper

def eager(f: Callable[..., Union[native.Promise, Coroutine[native.Promise, Any, Any]]]) -> Callable[..., Any]:
	"""
	Transforms a function returning a coroutine or promise into
	one that immediately executes via the main event loop.
	"""
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		return wait(f(*args, **kwargs))
	
	return wrapper