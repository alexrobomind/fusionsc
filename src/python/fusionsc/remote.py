"""Helpers to connect to remote instances"""

from . import service
from . import backends
from . import asnc

@asnc.asyncFunction
async def sshPublicKey(host, user, port = 21, pubKeyFile = None, privKeyFile = None, passPhrase = None):
	"""Creates an SSH session using public key authentication."""
	networkInterface = backends.localResources()
	
	connection = networkInterface.sshConnect(host, port).connection
	await connection.authenticateKeyFile(user, pubKeyFile, privKeyFile, passPhrase)
	return connection

@async.asyncFunction
async def connect(url : str, tunnel = None):
	"""Connects to the given URL, optionally using a network tunnel (e.g. an SSH connection)"""
	networkInterface = tunnel
	
	if networkInterface is None:
		networkInterface = backends.localResources()
	
	connection = (await networkInterface.connect(url)).connection
	backend = await connection.getRemote()
	return backend.remote