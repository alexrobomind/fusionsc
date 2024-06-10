"""Processing of DataRefs for distributed data (archives, ref publication, ref downloading)"""
from . import native
from . import backends
from . import service
from . import asnc
from . import capnp
from . import serialize

from .asnc import asyncFunction

from typing import Any, Union

def openArchive(filename: str) -> service.DataRef.Client:
	"""
	Opens the given archive file and provides a DataRef representing its root.
	
	Parameters:
		filename: File name of the archive file
	
	Returns:
		A DataRef pointing to the file root that can be shared or opened using the 'download' method.
	"""
	# A naive approach would be:
	#	return backends.localResources().openArchive(filename).ref
	#
	# However, direct requests do not carry the correct type
	# We therefore go to a two-step process where we open the file in this thread
	# to determine the type, then make a request to localResources to open it.
	# This happens in the C++ library
	return native.data.openArchive(filename, backends.localResources())
	

def publish(data: Any) -> service.DataRef.Client:
	"""
	Copies the provided data and provides a DataRef to the in-memory copy.
	
	Parameters:
		data: The information to store
	   
	Returns:
		A DataRef pointing to an in-memory copy of 'data'.
	"""
	inThreadRef = native.data.publish(serialize.wrap(data))
	cloneResult = backends.localResources().download(inThreadRef).pipeline.ref
	return cloneResult

@asyncFunction
async def download(ref: service.DataRef.Client) -> asnc.Future[Any]:
	"""
	Retrieves a local copy of the information stored in 'ref'. If possible, transfer of data will be avoided. The retrieved data
	are immutable and the backing storage is shared across all users in this process. If 'ref' was obtained from an archive file,
	the file contents will be directly mapped into memory
	
	Parameters:
		ref: The data ref to download onto
	
	Returns:
		The contents of the respective DataRef, interpreted as the appropriate type (either a DataReader if the ref holds raw binary
		data, or a subtype of CapabilityClient or StructReader typed to the appropriate Cap'n'proto schema).
	"""
	data = await native.data.downloadAsync(ref)
	return await serialize.unwrap.asnc(data)

@asyncFunction
def writeArchive(data: Any, filename: str) -> asnc.Future:
	"""
	Writes a copy of 'data' into a local archive file. All transitively contained DataRefs will be downloaded and a copy of them
	will be stored in this file. This ensures that an archive always contains a complete copy of all information needed to reconstruct
	its contents.
	"""
	if isinstance(data, capnp.CapabilityClient):
		ref = data
	else:
		ref = publish(data)
	
	return backends.localResources().writeArchive(filename, ref)

@asyncFunction
def readArchive(filename: str) -> asnc.Future[Any]:
	"""
	Opens the given archive file, maps the root node into memory and returns a typed view to the memory-mapped data.
	
	Parameters:
		filename: File name of the archive file
	
	Returns:
		The contents of the archive, interpreted as the appropriate type (either a DataReader if the ref holds raw binary
		data, or a subtype of CapabilityClient or StructReader typed to the appropriate Cap'n'proto schema).
	"""
	archiveRef = openArchive(filename)
	return download.asnc(archiveRef)

def upload(ref: service.DataRef.Client) -> service.DataRef.Client:
	# Obtain backend data service
	backend = backends.activeBackend()
	dataService = backend.dataService().pipeline.service
	
	# Start upload to remote end
	return dataService.cloneAllIntoMemory(ref).pipeline.ref