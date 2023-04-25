from . import native
from . import inProcess
from . import service
from . import asnc
from . import capnp

from .asnc import asyncFunction

from typing import Any, Union

@asyncFunction
def openArchive(filename: str) -> service.DataRef:
	"""
	Opens the given archive file and provides a DataRef representing its root.
	
	Parameters:
		filename: File name of the archive file
	
	Returns:
		A DataRef pointing to the file root that can be shared or opened using the 'download' method.
	"""
	return inProcess.localResources().openArchive(filename)

def publish(data: Union[capnp.DynamicStructReader, capnp.DataReader]) -> service.DataRef:
	"""
	Copies the provided data and provides a DataRef to the in-memory copy.
	
	Parameters:
		data: The information to store
	   
	Returns:
		A DataRef pointing to an in-memory copy of 'data'.
	"""
	inThreadRef = native.data.publish(data)
	print(type(inThreadRef))
	cloneResult = inProcess.localResources().download(inThreadRef)
	return cloneResult.ref

@asyncFunction
def download(ref: service.DataRef) -> asnc.Promise[Union[capnp.DynamicCapabilityClient, capnp.DynamicStructReader, capnp.DataReader]]:
	"""
	Retrieves a local copy of the information stored in 'ref'. If possible, transfer of data will be avoided. The retrieved data
	are immutable and the backing storage is shared across all users in this process. If 'ref' was obtained from an archive file,
	the file contents will be directly mapped into memory
	
	Parameters:
		ref: The data ref to download onto
	
	Returns:
		The contents of the respective DataRef, interpreted as the appropriate type (either a DataReader if the ref holds raw binary
		data, or a subtype of DynamicCapabilityClient or DynamicStructReader typed to the appropriate Cap'n'proto schema).
	"""
	return native.data.downloadAsync(ref)

@asyncFunction
def writeArchive(data: Union[capnp.DynamicStructReader, capnp.DynamicCapabilityClient], filename: str):
	"""
	Writes a copy of 'data' into a local archive file. All transitively contained DataRefs will be downloaded and a copy of them
	will be stored in this file. This ensures that an archive always contains a complete copy of all information needed to reconstruct
	its contents.
	"""
	if isinstance(data, capnp.DynamicCapabilityClient):
		ref = data
	else:
		ref = publish(data)
	
	return inProcess.localResources().writeArchive(filename, ref)

@asyncFunction
def readArchive(filename: str) -> asnc.Promise[Union[capnp.DynamicCapabilityClient, capnp.DynamicStructReader]]:
	"""
	Opens the given archive file, maps the root node into memory and returns a typed view to the memory-mapped data.
	
	Parameters:
		filename: File name of the archive file
	
	Returns:
		The contents of the archive, interpreted as the appropriate type (either a DataReader if the ref holds raw binary
		data, or a subtype of DynamicCapabilityClient or DynamicStructReader typed to the appropriate Cap'n'proto schema).
	"""
	return download.asnc(openArchive(filename))
