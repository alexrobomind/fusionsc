from . import native
from . import devices
from . import flt

from .asnc import run, asyncFunction, eager
from .resolve import importOfflineData, fieldResolvers, geometryResolvers, backupResolvers
from .native import delay

from typing import Optional

__all__ = [
	'local', 'tracer', 'run', 'wait', 'asyncFunction', 'eager', 'importOfflineData', 'fieldResolvers', 'geometryResolvers'
]

# Initialize event loop
native.startEventLoop()

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