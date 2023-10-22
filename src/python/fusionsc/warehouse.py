from . import service
from . import backends
from . import wrappers
from . import data

from typing import OneOf

class Object:
	def __init__(self, backend):
		self.backend = backend

# Types of objects that can be stored in a warehouse
Storable = OneOf[
	service.DataRef,
	wrappers.RefWrapper,
	capnp.StructReader,
	capnp.DataReader,
	Object # Warehouse objects can be stored inside each other IF they come from the same database
]

class Folder(Object):
	def __init__(self, backend):
		super().__init__(backend)
	
	@asyncFunction
	def ls(self, path: str = ""):
		"""
		Lists all entries currently present in this folder
		"""
		
		response = await self.backend.ls(path)
		return [str(x) for x in response.entries]
	
	@asyncFunction
	def getAll(self, path: str = ""):
		"""
		Returns a key-value mapping of all present entries
		"""
		
		response = await self.backend.getAll(path)
		return { str(entry.key) : _decode(entry.value) }
	
	@asyncFunction
	def get(self, path: str):
		"""
		Returns the object placed under the specified path
		"""
		
		response = await self.backend.get(path)
		return _decode(response)
	
	@asyncFunction
	def put(self, path: str, val: Storable):
		"""
		Places an object under the specified path. Creates parent directories as needed
		"""
		
		response = await self.backend.put(path, _encode(val))
		return _decode(response)
	
	@asyncFunction
	def rm(self, path: str):
		"""
		Removes the specified path
		"""
		
		await self.backend.rm(path)
	
	@asyncFunction
	def createFile(self, path: str = ""):
		"""
		Creates a new database file at the given path.
		
		If no path is specified, instead creates a temporary in-memory file that will
		be linked into the database directory as soon as it is stored in a parent object.
		"""
		
		response = await self.backend.createFile(path)
		return File(response.file)
	
	def freeze(self, path: str = ""):
		"""
		Returns a frozen version of the directory structure contained inside this folder
		tree. The structure returns a snapshot of all directories and files. Actual stored
		data will be linked as DataRefs.
		"""
		
		pipeline = self.backend.freeze(path)
		return wrappers.RefWrapper(pipeline.ref)

class File(Object):
	def __init__(self, backend):
		super().__init__(backend)
	
	@asyncFunction
	def get(self):
		"""
		Accesses the contents of the target file (by reference)
		"""
		
		response = await self.backend.getAny()
		return _decode(response)
	
	@asyncFunction
	def read(file):
		"""
		Downloads the contents of the target file (assuming it's
		a DataRef)
		"""
		response = self.backend.get()
		return await data.download(response.asGeneric)
	
	@asyncFunction
	def put(self, val: Storable):
		response = await self.backend.put(_encode(val))
		return _decode(response)

class Unknown(Folder, File, wrappers.RefWrapper):
	"""
	In some cases we don't know the exact type of object stored.
	This class subclasses all possible interfaces.
	"""
	def __init__(self, backend):
		super().__init__(backend)
		wrappers.RefWrapper.__init__(self, backend)

def _encode(obj: Storable) -> service.CapabilityClient):
	if isinstance(obj, service.DataRef):
		return obj
	
	if isinstance(obj, wrappers.RefWrapper):
		return obj.ref
		
	if isinstance(obj, capnp.StructReader) or
		isinstance(obj, capnp.DataReader) or
		isinstance(obj, capnp.StructBuilder) or
		isinstance(obj, capnp.DataBuilder):
		return data.publish(obj)
	
	if isinstance(obj, Object):
		return obj.backend
	
	raise ValueError("Object is not storable in warehouse")

def _decode(obj: service.warehouse.StoredObject.Reader):
	which = obj.which()
	
	# Cases where the object type is already known
	if which == "folder":
		return Folder(obj.folder)
	
	if which == "file":
		return File(obj.file)
	
	if which == "dataRef":
		return wrappers.RefWrapper(obj.dataRef.asRef)
	
	# Cases where the object type is not known (delayed / failed resolution)
	return Unknown(obj.asGeneric)