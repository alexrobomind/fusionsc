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
	"""Connects a thread to the in-process worker"""
	asnc.startEventLoop()
	_threadLocal.localResources = inProcessWorker.connect()
	_threadLocal.root = _threadLocal.localResources.root().root
	
def disconnectLocal():
	del _threadLocal.localResources
	del _threadLocal.root

def localResources():
	return _threadLocal.localResources

def localBackend():
	assert isLocalConnected(), """
		This thread is not connected to the local backend. Please call fusionsc.backends.connectLocal()
		(which is automatically done for the main thread), wrap your code in 'with fusionsc.backends.useBackend(...):',
		or call 'fusionsc.backends.alwaysUseBackend(...):'
	"""
	
	return _threadLocal.root

def isLocalConnected():
	return hasattr(_threadLocal, "root")

def activeBackend():
	if hasattr(_threadLocal, "active"):
		return _threadLocal.active
	
	return localBackend()

@contextlib.contextmanager
def useBackend(newBackend):
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
	_threadLocal.active = newBackend