import functools
import warnings

def untested(f):
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		warnings.warn("The function {f.__module__}.{f.__qualname__} has not yet been properly tested. Please report any errors you find.")
		return f(*args, **kwargs)
	
	return wrapper

def unstableApi(f):
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		warnings.warn(f"The function {f.__module__}.{f.__qualname__} is part of the unstable API. It might change or get removed in the near future. While unlikely, it might also not be compatible across client/server versions.")
		return f(*args, **kwargs)
	
	return wrapper
