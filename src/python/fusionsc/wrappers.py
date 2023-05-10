from .asnc import asyncFunction
from . import data, capnp
import functools
import warnings

class StructWrapperBase:
	def __init__(self, val = None, msgSize = 1024, byReference = False):
		if byReference and val is not None:
			self._val = val
			
		self._val = self.type.newMessage(val, msgSize)
	
	@property
	def data(self):
		return self._val
	
	@data.setter
	def data(self, newVal):
		self._val = self.type.newMessage(newVal)
	
	@asyncFunction
	async def save(self, filename):
		await data.writeArchive.asnc(self.data, filename)
	
	@asyncFunction
	@classmethod
	async def load(cls, filename):
		newData = await data.readArchive.asnc(filename)
		return cls(newData, byReference = True)
	
	def ptree(self):
		import printree
		printree.ptree(self.data)
	
	def graph(self, **kwargs):
		return capnp.visualize(self.field, **kwargs)
	
	def toYaml(self):
		return capnp.toYaml(self.data)
			
	def __repr__(self):
		return str(self.data)

def structWrapper(serviceType):
	class Wrapper(StructWrapperBase):
		type = serviceType
	
	return Wrapper

def untested(f):
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		warnings.warn("This function has not yet been tested. Please report any errors you find")
		return f(*args, **kwargs)
	
	return wrapper

def unstableAPI(f):
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		warnings.warn("This function is part of the unstable API. It might change or get removed in the near future.")
		return f(*args, **kwargs)
	
	return wrapper