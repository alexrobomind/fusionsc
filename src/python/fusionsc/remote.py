"""Helpers to connect to remote instances"""

from . import service
from . import backends
from . import asnc
from . import capnp

from .asnc import asyncFunction
from ._api_markers import unstableApi, untested

from typing import Optional, Any, Union

class OpenPort:
	"""
	Represents an open port on a local (or remote)
	network interface.
	"""
	
	_backend: service.NetworkInterface.OpenPort
	
	def __init__(self, newBackend):
		self._backend = newBackend
	
	@asyncFunction
	async def getPort(self) -> int:
		info = await self._backend.getInfo()
		return info.port
	
	@asyncFunction
	async def drain(self):
		"""
		Wait until all open connections are closed
		"""
		await self._backend.drain()
	
	@asyncFunction
	async def stopListening(self):
		"""
		Stop accepting requests on this port.
		
		Does not terminate existing connections
		"""
		await self._backend.stopListening()
	
	@asyncFunction
	async def closeAll(self):
		"""
		Force-close all existing connections on this port
		"""
		await self._backend.closeAll()
	

@asyncFunction
@unstableApi
async def sshPublicKey(host, user, port = 21, pubKeyFile = None, privKeyFile = None, passPhrase = None) -> service.NetworkInterface:
	"""
	Creates an SSH session using public key authentication.
	"""
	networkInterface = backends.localResources()
	
	connection = networkInterface.sshConnect(host, port).pipeline.connection
	await connection.authenticateKeyFile(user, pubKeyFile, privKeyFile, passPhrase)
	return connection

@asyncFunction
async def connect(url: str, tunnel: "Optional[None, service.NetworkInterface]" = None, ServiceType = service.RootService) -> Any:
	"""
	Connects to the given URL, optionally using a network tunnel (e.g. an SSH connection)
	"""
	networkInterface = tunnel
	
	if networkInterface is None:
		networkInterface = backends.localResources()
	
	connection = (await networkInterface.connect(url)).connection
	backend = await connection.getRemote()
	
	return backend.remote.castAs_(ServiceType)

@asyncFunction
@untested
async def serve(target: capnp.Capability.Client, host: str = "0.0.0.0", port: Optional[int] = None, tunnel: "Optional[service.NetworkInterface]" = None) -> OpenPort:
	"""
	Serves the given object (or the active backend) on the given host and port, optionally over the specified connection
	"""
	
	networkInterface = tunnel
	
	if networkInterface is None:
		networkInterface = backends.localResources()
	
	serveResponse = await networkInterface.serve(
		host, port if port is not None else 0, target
	)
	
	return OpenPort(serveResponse.openPort)
