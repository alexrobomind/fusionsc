# Load native library
from . import native

# Add sub-packages from native library
from .native import kj
from .native import capnp
from .native import schema
from .native import service

# Load asynchronous subprocessing module
# (this also imports relevant portions of native.asnc)
from . import asnc

# Connect to local worker
from . import inProcess

# Load auxiliary modules
from . import data
from . import worker
from . import ipython_integration
from . import resolve

# Load core services
from . import flt
from . import geometry

# Load device-specifics
from . import devices

# Import some nice helpers into the root namespace
from .asnc import run, asyncFunction, wait, Promise, delay
from .resolve import importOfflineData

from typing import Optional

import threading

__all__ = [
	'run', 'asyncFunction', 'wait', 'importOfflineData', 'delay', 'Promise', 'MagneticConfig'
]

def tracer(backend: Optional[native.RootService] = None) -> flt.FLT:
	"""
	Creates a new field line tracer backed by an FSC service. If
	no backend is specified, creates a local backend.
	"""
	
	if backend is None:
		backend = local()
			
	return flt.FLT(backend)
	
	
	