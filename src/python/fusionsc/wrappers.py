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

def untested(f):
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		warnings.warn("This function has not yet been tested. Please report any errors you find")
		return f(*args, **kwargs)
	
	return wrapper

def unstableApi(f):
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		warnings.warn(f"The function {f.__module__}.{f.__qualname__} is part of the unstable API. It might change or get removed in the near future. While unlikely, it might also not be compatible across client/server versions.")
		return f(*args, **kwargs)
	
	return wrapper
