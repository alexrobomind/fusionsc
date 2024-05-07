"""Manages the active backend to use for calculation"""
from . import asnc
from . import native
from . import service
from . import config

import contextlib
import contextvars
import threading

# Initialize event loop for main thread
asnc.startEventLoop()

# Create a new in-process worker living in a separate thread
inProcessWorker = native.LocalRootServer()

#_localResources = contextvars.ContextVar("fusionsc.backends._localResources", default = (None, None))
_localResources = asnc.EventLoopLocal(default = None)
_currentBackend = contextvars.ContextVar("fusionsc.backends._currentBackend", default = (None, None))

def _threadId():
	return threading.get_ident()

def connectLocal():
	"""Connects a thread to the in-process worker. Automatically called for main thread."""
	asnc.startEventLoop()
	_localResources.value = inProcessWorker.connect()
	
def disconnectLocal():
	"""Disconnects a thread from the in-process worker"""
	del _localResources.value

def isLocalConnected() -> bool:
	"""
	Checks whether the thread is connected to the in-process worker
	"""
	return _localResources.value is not None

def localResources() -> service.LocalResources.Client:
	"""
	Provides access to the in-process worker's "LocalResources" service interface.
	
	This interface can be used to access several privileged functions (like network setup and file
	system access) that are not available for remote clients.
	"""
	if not isLocalConnected():
		connectLocal()
	
	return _localResources.value

def localBackend() -> service.RootService.Client:
	"""
	Returns the backend corresponding to the in-process worker. Requires the active thread
	to be connected to it.
	"""
	return localResources().root().pipeline.root

@asnc.asyncFunction
async def reconfigureLocalBackend(config: service.LocalConfig.ReaderOrBuilder):
	"""
	Reconfigues the local backend to the given configuration.
	"""
	await localResources().configureRoot(config)

def activeBackend() -> service.RootService.Client:
	"""
	Returns the current thread's active backend for calculations. This is the inner-most
	active useBackend call, falling back to the in-process worker if no other backend
	is in use
	"""
	threadId, cb = _currentBackend.get()
	
	if cb is not None:
		assert threadId == _threadId(), "The current backend was passed from a different thread. This is not supported"
		return cb
	
	# If backend not set, check if it is set in configuration
	confCtx = config.context()
	if _currentBackend in confCtx:
		_, cb = confCtx[_currentBackend]
		return cb
	
	return localBackend()

@contextlib.contextmanager
def useBackend(newBackend: service.RootService.Client):
	"""
	Temporarily overrides the active backend to use for calculations.
	
	Example usage::
		import fusionsc as fsc
		...
		newBackend = ...
		with fsc.backends.useBackend(newBackend):
			... Calculation code ...
	"""
	token = _currentBackend.set((_threadId(), newBackend))
	yield newBackend
	_currentBackend.reset(token)

def alwaysUseBackend(newBackend: service.RootService.Client):
	"""
	Permanently installs a backend as the default for this thread.
	
	Note that exiting newBackend(...) scopes also removes the backend installed by this function if
	it was installed inside.
	"""
	_currentBackend.set((_threadId(), newBackend))

@asnc.asyncFunction
async def backendInfo() -> service.NodeInfo.Reader:
	"""Returns information about the currently active backend"""
	return await activeBackend().getInfo()
