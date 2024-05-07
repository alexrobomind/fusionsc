"""
The FusionSC package for fusion-related scientific computations.
"""

# Load native library
from . import native

# Add sub-packages from native library
from .native import kj
from .native import capnp
from .native import loader
from .native import efit

# Load service definitions
from . import service

# Load helpers for pickling
from . import pickle_support

# Load asynchronous subprocessing module
# (this also imports relevant portions of native.asnc)
from . import asnc

# Load user configuration
from . import config

# Connect to local worker
from . import backends
backends.connectLocal()

# Load remote connection module
from . import remote

# Load auxiliary modules
from . import data
from . import resolve
from . import wrappers
from . import warehouse
from . import structio

# Load core services
from . import magnetics
from . import flt
from . import geometry
from . import hfcam

from . import hint
from . import vmec

# Load device-specifics
from . import devices

# Import some nice helpers into the root namespace
from typing import Optional

from . import export

import threading

__all__ = [
	'native',
	'kj', 'capnp', 'service', 'efit', 'hint', 'vmec',
	'asnc', 'backends', 'data', 'ipython_integration', 'resolve',
	'magnetics', 'flt', 'hfcam', 'devices', 'export', 'structio',
	'geometry'
]
	
