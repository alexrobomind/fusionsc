from . import native
from . import devices
from . import flt

from .asnc import run, asyncFunction

from typing import Optional


# Initialize event loop
native.startEventLoop()

def local() -> native.RootService:
	"""
	Creates a local instance of the FSC services
	"""
	
	return native.connectSameThread()

def tracer(backend: Optional[native.RootService] = None, grid: Optional[native.ToroidalGrid] = None) -> flt.FLT:
	"""
	Creates a new field line tracer backed by an FSC service. If
	no backend is specified, creates a local backend.
	"""
	
	if backend is None:
		backend = local()
	
	if grid is None:
		grid = devices.w7x.defaultGrid()
		
	return flt.FLT(backend, grid)