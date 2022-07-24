from . import native
from . import devices
from . import flt
from . import asnc
from . import data

from .native import kj
from .native import capnp

from .asnc import run, asyncFunction, eager, wait, Promise
from .resolve import importOfflineData

from .native.timer import delay

from typing import Optional

__all__ = [
	'run', 'asyncFunction', 'eager', 'wait', 'importOfflineData', 'delay', 'Promise'
]

# Initialize event loop for main thread
asnc.startEventLoop()

def local() -> native.RootService:
	"""
	Creates a local instance of the FSC services
	"""
	
	return native.connectSameThread()

def tracer(backend: Optional[native.RootService] = None) -> flt.FLT:
	"""
	Creates a new field line tracer backed by an FSC service. If
	no backend is specified, creates a local backend.
	"""
	
	if backend is None:
		backend = local()
			
	return flt.FLT(backend)