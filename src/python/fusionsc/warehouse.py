"""
This module can be used to access 'Warehouses', remotely accessible mutable object stores.
"""

from . import capnp
from . import service
from . import backends
from . import wrappers
from . import data

from .wrappers import asyncFunction

from typing import Union

class Object:
	def __init__(self, backend):
		self.backend = backend

# Types of objects that can be stored in a warehouse
Storable = Union[
	service.DataRef.Client,
	wrappers.RefWrapper,
	capnp.StructReader,
	capnp.DataReader,
	Object # Warehouse objects can be stored inside each other IF they come from the same database
]

@asyncFunction
async def lsRemote():
	"""Lists the warehouses available through the current (possibly remote) backend"""
	response = await backends.activeBackend().listWarehouses()
	return [str(name) for name in response.names]

@asyncFunction
async def openRemote(name: str):
	"""Opens the named warehouse exposed by the current backend"""
	response = await backends.activeBackend().getWarehouse(name)
	return Folder(response.warehouse)

@asyncFunction
async def open(url: str):
	"""
	Opens a warehouse via a URL.
	
	Supported url schemes are:
		'sqlite': SQLite database on current file system
		'ws' or 'http': Remote warehouse server that can be connected to via network.
		'remote': Looks up a named warehouse exposed by the active backend (remote:myrepo)
	"""
	if url.startswith('remote:'):
		# URL fragments indicate subfolders
		# We still want to honor this correctly
		fragment = None
		if "#" in url:
			url, fragment = url.split("#")
			
		name = url[7:].split('?')[0]
		
		remote = await openRemote.asnc(name)
		
		if fragment is not None:
			remote = remote.get(fragment)
		
		return remote
	
	response = await backends.localResources().openWarehouse(url)
	return _decode(response.storedObject)

class Folder(Object):
	"""
	A mutable folder inside a warehouse
	"""
	
	def __init__(self, backend):
		super().__init__(backend)
	
	@asyncFunction
	async def ls(self, path: str = ""):
		"""
		Lists all entries currently present in this folder
		"""
		
		response = await self.backend.ls(path)
		return [str(x) for x in response.entries]
	
	@asyncFunction
	async def getAll(self, path: str = ""):
		"""
		Returns a key-value mapping of all present entries
		"""
		
		response = await self.backend.getAll(path)
		return { str(entry.key) : _decode(entry.value) for entry in response.entries }
	
	@asyncFunction
	async def get(self, path: str):
		"""
		Returns the object placed under the specified path
		"""
		
		response = await self.backend.get(path)
		return _decode(response)
	
	@asyncFunction
	async def put(self, path: str, val: Storable):
		"""
		Places an object under the specified path. Creates parent directories as needed
		"""
		
		response = await self.backend.put(path, _encode(val))
		return _decode(response)
	
	@asyncFunction
	async def rm(self, path: str):
		"""
		Removes the specified path
		"""
		
		await self.backend.rm(path)
	
	@asyncFunction
	async def createFile(self, path: str = ""):
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
	async def get(self):
		"""
		Accesses the contents of the target file (by reference)
		"""
		
		response = await self.backend.getAny()
		return _decode(response)
	
	@asyncFunction
	async def read(file):
		"""
		Downloads the contents of the target file (assuming it's
		a DataRef)
		"""
		return await data.download.asnc(self.backend.get().ref)
	
	@asyncFunction
	async def put(self, val: Storable):
		response = await self.backend.put.asnc(_encode(val))
		return _decode(response)

class Unknown(Folder, File, wrappers.RefWrapper):
	"""
	In some cases we don't know the exact type of object stored.
	This class subclasses all possible interfaces.
	"""
	def __init__(self, backend):
		super().__init__(backend)
		wrappers.RefWrapper.__init__(self, backend)

def _encode(obj: Storable) -> capnp.CapabilityClient:
	if isinstance(obj, service.DataRef.Client):
		return obj
	
	if isinstance(obj, Object):
		return obj.backend
	
	if isinstance(obj, wrappers.RefWrapper):
		return obj.ref
		
	return data.publish(obj)

def _decode(obj: service.Warehouse.StoredObject.ReaderOrBuilder):
	which = obj.which_()
	
	# Cases where the object type is already known
	if which == "folder":
		return Folder(obj.folder)
	
	if which == "file":
		return File(obj.file)
	
	if which == "dataRef":
		return wrappers.RefWrapper(obj.dataRef.asRef)
	
	# Cases where the object type is not known (delayed / failed resolution)
	return Unknown(obj.asGeneric)
