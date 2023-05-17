from .asnc import asyncFunction
from . import data, capnp
import functools
import warnings

class StructWrapperBase:
	def __init__(self, val = None, msgSize = 1024, byReference = False):
		"""
		Creates a new instance by either:
		- Allocating a new message of size msgSize (if val is None)
		- Storing a reference to the target (if byReference is set)
		- Copying the message from the given target (accepts Cap'n'proto messages,
			YAML strings, lists, dicts, and NumPy arrays)
		"""
		if byReference and val is not None:
			self._val = val
			
		self._val = self.type.newMessage(val, msgSize)
	
	@property
	def data(self):
		"""
		Underlying Cap'n'proto struct (usually a builder type).
		"""
		return self._val
	
	@data.setter
	def data(self, newVal):
		self._val = self.type.newMessage(newVal)
	
	@asyncFunction
	async def save(self, filename):
		"""
		Saves the object into an archive file (including all referenced data)
		"""
		await data.writeArchive.asnc(self.data, filename)
	
	@asyncFunction
	@classmethod
	async def load(cls, filename):
		"""
		Loads an instance of this class from a previously written archive file.
		"""
		newData = await data.readArchive.asnc(filename)
		return cls(newData, byReference = True)
	
	def ptree(self):
		"""
		Prints a tree version of this object. Requires the printree library.
		"""
		import printree
		printree.ptree(self.data)
	
	def graph(self, **kwargs):
		"""
		Visualizes the contents in a graph structure. Requires graphviz
		"""
		return capnp.visualize(self.field, **kwargs)
	
	def toYaml(self):
		"""
		Prints the contents as block-structured YAML
		"""
		return capnp.toYaml(self.data)
			
	def __repr__(self):
		"""
		Single-line YAML representation of the contents.
		"""
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