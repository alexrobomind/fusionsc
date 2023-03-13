from . import asnc

# Initialize event loop for main thread
asnc.startEventLoop()

inProcessWorker = native.LocalRootServer()

_threadLocal = threading.local()

def connect():
    _threadLocal.localResources = inProcessWorker.connect()
    _threadLocal.root = threadLocal.localResources.root().root
    
def disconnect():
    del _threadLocal.localResources
    del _threadLocal.root

def localResources():
    return _threadLocal.localResources

def root():
    return _threadLocal.root

connect()