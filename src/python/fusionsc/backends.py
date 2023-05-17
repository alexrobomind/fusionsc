"""Manages the active backend to use for calculation"""
from . import asnc
from . import native

import contextlib

# Initialize event loop for main thread
asnc.startEventLoop()

# Create a new in-process worker living in a separate thread
inProcessWorker = native.LocalRootServer()

import threading
_threadLocal = threading.local()

def connectLocal():
	"""Connects a thread to the in-process worker. Automatically called for main thread."""
	asnc.startEventLoop()
	_threadLocal.localResources = inProcessWorker.connect()
	_threadLocal.root = _threadLocal.localResources.root().root
	
def disconnectLocal():
	"""Disconnects a thread from the in-process worker"""
	del _threadLocal.localResources
	del _threadLocal.root

def localResources():
	"""
	Provides access to the in-process worker's "LocalResources" service interface.
	
	This interface can be used to access several privileged functions (like network setup and file
	system access) that are not available for remote clients.
	"""
	return _threadLocal.localResources

def localBackend():
	"""
	Returns the backend corresponding to the in-process worker. Requires the active thread
	to be connected to it.
	"""
	assert isLocalConnected(), """
		This thread is not connected to the local backend. Please call fusionsc.backends.connectLocal()
		(which is automatically done for the main thread), wrap your code in 'with fusionsc.backends.useBackend(...):',
		or call 'fusionsc.backends.alwaysUseBackend(...):'
	"""
	
	return _threadLocal.root

def isLocalConnected():
	"""
	Checks whether the thread is connected to the in-process worker
	"""
	return hasattr(_threadLocal, "root")

def activeBackend():
	"""
	Returns the current thread's active backend for calculations. This is the inner-most
	active useBackend call, falling back to the in-process worker if no other backend
	is in use
	"""
	if hasattr(_threadLocal, "active"):
		return _threadLocal.active
	
	return localBackend()

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
	if hasattr(_threadLocal, "active"):
		prevBackend = _threadLocal.active
	else:
		prevBackend = None
	
	_threadLocal.active = newBackend
	
	yield newBackend
	
	if prevBackend is None:
		del _threadLocal.active
	else:
		_threadLocal.active = prevBackend


def alwaysUseBackend(newBackend):
	"""
	Permanently installs a backend as the default for this thread.
	
	Note that exiting newBackend(...) scopes also removes the backend installed by this function if
	it was installed inside.
	"""
	_threadLocal.active = newBackend