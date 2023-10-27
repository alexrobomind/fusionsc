"""Manages the active backend to use for calculation"""
from . import asnc
from . import native
from . import service

import contextlib
import contextvars

# Initialize event loop for main thread
asnc.startEventLoop()

# Create a new in-process worker living in a separate thread
inProcessWorker = native.LocalRootServer()

import threading
_threadLocal = threading.local()

_currentBackend = contextvars.ContextVar("fusionsc.backends._currentBackend", default = None)

def connectLocal():
	"""Connects a thread to the in-process worker. Automatically called for main thread."""
	asnc.startEventLoop()
	_threadLocal.localResources = inProcessWorker.connect()
	
def disconnectLocal():
	"""Disconnects a thread from the in-process worker"""
	del _threadLocal.localResources

def localResources():
	"""
	Provides access to the in-process worker's "LocalResources" service interface.
	
	This interface can be used to access several privileged functions (like network setup and file
	system access) that are not available for remote clients.
	"""
	if not isLocalConnected():
		connectLocal()
	
	return _threadLocal.localResources

def localBackend():
	"""
	Returns the backend corresponding to the in-process worker. Requires the active thread
	to be connected to it.
	"""
	return localResources().root().pipeline.root

@asnc.asyncFunction
async def reconfigureLocalBackend(config):
	"""
	Reconfigues the local backend to the given configuration.
	"""
	await localResources().configureRoot(config)

def isLocalConnected():
	"""
	Checks whether the thread is connected to the in-process worker
	"""
	return hasattr(_threadLocal, "localResources")

def activeBackend():
	"""
	Returns the current thread's active backend for calculations. This is the inner-most
	active useBackend call, falling back to the in-process worker if no other backend
	is in use
	"""
	cb = _currentBackend.get()
	
	if cb is None:
		cb = localBackend()
	
	return cb

@contextlib.contextmanager
def useBackend(newBackend):
	"""
	Temporarily overrides the active backend to use for calculations.
	
	Example usage::
		import fusionsc as fsc
		...
		newBackend = ...
		with fsc.backends.useBackend(newBackend):
			... Calculation code ...
	"""
	token = _currentBackend.set(newBackend)
	yield newBackend
	_currentBackend.reset(token)

def alwaysUseBackend(newBackend):
	"""
	Permanently installs a backend as the default for this thread.
	
	Note that exiting newBackend(...) scopes also removes the backend installed by this function if
	it was installed inside.
	"""
	_currentBackend.set(newBackend)