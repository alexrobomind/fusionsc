from .asnc import asyncFunction
from . import data, capnp
import functools
import warnings
import operator

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
	def _fusionsc_wraps(self):
		return self._val
	
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
	
	@asyncFunction
	async def cache(self, filename):
		from pathlib import Path
		p = Path(filename)
		
		# Write out representation to file
		if not p.exists():
			await self.save.asnc(str(filename))
		
		# Load again
		return await self.load.asnc(str(filename))
	
	@asyncFunction
	async def upload(self):
		# Start upload
		remote = data.upload(data.publish(self.data))
		
		downloaded = await data.download.asnc(remote)
		return self.__class__(downloaded)
	
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
	
	def toYaml(self, flow = False):
		"""
		Prints the contents as block-structured YAML
		"""
		return self.data.toYaml_(flow)
			
	def __repr__(self):
		"""
		Single-line YAML representation of the contents.
		"""
		return str(self.data)

def structWrapper(serviceType):
	class Wrapper(StructWrapperBase):
		type = serviceType
		
		def __copy__(self):
			return Wrapper(self.data)
		
		def __deepcopy__(self, memo):
			return Wrapper(self.data)
	
	return Wrapper

class RefWrapper:
	def __init__(self, dataOrRef):
		if not isinstance(dataOrRef, capnp.CapabilityClient):
			dataOrRef = data.publish(dataOrRef)
			
		self._ref = dataOrRef
	
	@property
	def _fusionsc_wraps(self):
		return self._ref
	
	@property
	def ref(self):
		"""
		The underlying service.DataRef object.
		"""
		return self._ref
	
	def __await__(self):
		return self.ref.__await__()
	
	@asyncFunction
	async def download(self):
		"""
		Downloads (once it is available) the referenced data. Returns a reader for the result.
		
		*Note*: Data pointed to by DataRefs are immutable. If you want to modify the stored data,
		        clone the results with its `clone_()` method (including the trailing underscore).
		"""
		return await data.download.asnc(self.ref)
	
	@asyncFunction
	async def save(self, filename):
		"""
		Saves the referenced data into an archive file (including all referenced data)
		"""
		await data.writeArchive.asnc(self.ref, filename)
	
	@classmethod
	def load(cls, filename):
		"""
		Loads an instance of this class from a previously written archive file.
		"""
		return cls(data.openArchive(filename))

class LazyObject:
	__slots__ = ("_func", "_backend", "_evaluated")
	
	def _newProxy(name):
		def inner(self, *args, **kwargs):
			return getattr(self._delegate, name)(*args, **kwargs)
		
		return inner
	
	def __init__(self, func):
		self._func = func
		self._evaluated = False
	
	@property
	def _delegate(self):
		if not self._evaluated:
			self._backend = self._func()
			self._func = None
			self._evaluated = True
		
		return self._backend
	
	def __getattr__(self, name):
		return getattr(self._delegate, name)
	
	def __setattr__(self, name, val):
		if name in ("_func", "_backend", "_evaluated"):
			return super().__setattr__(name, val)
		
		return setattr(self._delegate, name, val)
	
	def __delattr__(self, name):
		if name in ("_func", "_backend", "_evaluated"):
			raise TypeError("Builtin attributes of LazyObject can't be deleted")
		return delattr(self._delegate, name)
	
	@property
	def __class__(self):
		return self._delegate.__class__
	
	def __dir__(self):
		return dir(self._delegate)
	
	def __hash__(self):
		return hash(self._delegate)
	
	def __bool__(self):
		return bool(self._delegate)
	
	def __repr__(self):
		return repr(self._delegate)
	
	def __str__(self):
		return str(self._delegate)
	
	def __complex__(self):
		return complex(self._delegate)
	
	def __float__(self):
		return float(self._delegate)
	
	def __int__(self):
		return int(self._delegate)
		
# Add various descriptors to the lazy array

class ProxyDescriptor:
	def __init__(self, name):
		self.name = name
		
	def __get__(self, obj, objtype=None):
		if obj is not None:
			return getattr(obj._delegate, self.name)
		return getattr(objtype, self.name)

class NIProxyDescriptor:
	def __init__(self, name):
		self.name = name
		
	def __get__(self, obj, objtype=None):
		if obj is not None:
			if not hasattr(obj._delegate, self.name):
				def ni(*args, **kwargs):
					return NotImplemented
				return ni
				
			return getattr(obj._delegate, self.name)
		return getattr(objtype, self.name)

for attr in [
	"__bytes__", "__format__",
	"__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__",
	"__getitem__", "__setitem__", "__delitem__",
	"__iter__", "__reversed__", "__contains__", "__len__",
	"__round__", "__trunc__", "__floor__", "__ceil__"
]:
	setattr(LazyObject, attr, ProxyDescriptor(attr))

for op in [
	"add", "sub", "mul", "floordiv", "truediv", "mod", "divmod",
	"matmul", "mod", "divmod", "pow", "lshift", "rshift", "and", "xor"
	"or"]:
	for attr in [f"__{op}__", "__r{op}__", "__i{op}__"]:
		setattr(LazyObject, attr, NIProxyDescriptor(attr))	
		

class LazyArray(LazyObject):
	def __init__(self, x):
		import numpy as np
		super().__init__(lambda: np.asarray(x))
		