from . import service
from . import backends
from . import asnc

@asnc.asyncFunction
async def sshPublicKey(host, user, port = 21, pubKeyFile = None, privKeyFile = None, passPhrase = None):
	networkInterface = backends.localResources()
	
	connection = networkInterface.sshConnect(host, port).connection
	await connection.authenticateKeyFile(user, pubKeyFile, privKeyFile, passPhrase)

@async.asyncFunction
async def connect(url : str, tunnel = None)
	networkInterface = tunnel
	
	if networkInterface is None:
		networkInterface = backends.localResources()
	
	connection = (await networkInterface.connect(url)).connection
	return (await connection.getRemote()).remote