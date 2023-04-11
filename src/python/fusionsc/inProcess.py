from . import asnc
from . import native

# Initialize event loop for main thread
asnc.startEventLoop()

# Create a new in-process worker living in a separate thread
inProcessWorker = native.LocalRootServer()

import threading
_threadLocal = threading.local()

def connect():
    """Connects a thread to the in-process worker"""
    _threadLocal.localResources = inProcessWorker.connect()
    _threadLocal.root = _threadLocal.localResources.root().root
    
def disconnect():
    del _threadLocal.localResources
    del _threadLocal.root

def localResources():
    return _threadLocal.localResources

def root():
    return _threadLocal.root

connect()