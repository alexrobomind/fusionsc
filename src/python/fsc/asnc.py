from . import native

from typing import Coroutine, Callable, Any

import functools

def run(coroutine: Coroutine[native.Promise, Any, Any]) -> native.Promise:
	"""
	Transforms a coroutine result (that can in turn await promises)
	into a promise
	"""
	return native.run(coroutine)

def asyncFunction(f: Callable[..., Coroutine[native.Promise, Any, Any]]) -> Callable[..., native.Promise]:
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		return run(f(*args, **kwargs))
	
	return wrapper

