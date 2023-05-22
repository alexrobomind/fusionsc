"""Helpers to connect to remote instances"""

from . import service
from . import backends
from . import asnc

class OpenPort:
	_backend: service.NetworkInterface.OpenPort
	
	def __init__(self, newBackend):
		self._backend = newBackend
	
	@asnc.asyncFunction
	async def getPort(self):
		info = await self._backend.getInfo()
		return info.port
	
	@asnc.asyncFunction
	async def drain(self):
		await self._backend.drain()
	
	@asnc.asyncFunction
	async def stopListening(self):
		await self._backend.stopListening()
	
	@asnc.asyncFunction
	async def closeAll(self):
		await self._backend.closeAll()
	

@asnc.asyncFunction
async def sshPublicKey(host, user, port = 21, pubKeyFile = None, privKeyFile = None, passPhrase = None):
	"""Creates an SSH session using public key authentication."""
	networkInterface = backends.localResources()
	
	connection = networkInterface.sshConnect(host, port).connection
	await connection.authenticateKeyFile(user, pubKeyFile, privKeyFile, passPhrase)
	return connection

@asnc.asyncFunction
async def connect(url : str, tunnel = None, ServiceType = service.RootService):
	"""Connects to the given URL, optionally using a network tunnel (e.g. an SSH connection)"""
	networkInterface = tunnel
	
	if networkInterface is None:
		networkInterface = backends.localResources()
	
	connection = (await networkInterface.connect(url)).connection
	backend = await connection.getRemote()
	
	return ServiceType.castAs(backend.remote)

@asnc.asyncFunction
async def serve(target, host = "0.0.0.0", port = None, tunnel = None):
	"""Serves the given object (or the active backend) on the given host and port, optionally over the specified connection"""
	networkInterface = tunnel
	
	if networkInterface is None:
		networkInterface = backends.localResources()
	
	serveResponse = await networkInterface.serve(
		host, port if port is not None else 0, target
	)
	
	return OpenPort(serveResponse.openPort)