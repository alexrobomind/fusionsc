"""Resolution helpers to obtain computable fields and geometries from high-level descriptions"""

from . import native, data
from .asnc import asyncFunction, startEventLoop

from .native.devices import w7x as cppw7x
from .native.devices import jtext as cppjtext
import contextlib

# Ensure event loop is running
startEventLoop()

fieldResolvers = [
	cppw7x.fieldResolver(),
	cppjtext.fieldResolver()
]

geometryResolvers = [
	cppw7x.geometryResolver(),
	cppjtext.geometryResolver()
]

def importOfflineData(filename: str):
	"""
	Loads the data contained in the given offline archives and uses them to
	perform offline resolution.
	"""
	offlineData = data.openArchive(filename)
	
	# Install offline resolvers
	fieldResolvers.append(native.offline.fieldResolver(offlineData))
	geometryResolvers.append(native.offline.geometryResolver(offlineData))

@asyncFunction
async def resolveField(field, followRefs: bool = False):		
	for r in fieldResolvers:
		field = await r.resolveField(field, followRefs)
		
	return field

@asyncFunction
async def resolveFilament(filament, followRefs: bool = False):		
	for r in fieldResolvers:
		filament = await r.resolveFilament(filament, followRefs)
		
	return filament
	
@asyncFunction
async def resolveGeometry(geometry, followRefs: bool = False):		
	for r in geometryResolvers:
		geometry = await r.resolveGeometry(geometry, followRefs)
		
	return geometry

@contextlib.contextmanager
def backupResolvers():
	"""A context manager that, upon exiting, restores the resolver lists to their state when entered"""
	global fieldResolvers
	global geometryResolvers
	
	backupFR = fieldResolvers.copy()
	backupGR = geometryResolvers.copy()
	
	try:
		yield None
	finally:
		fieldResolvers = backupFR
		geometryResolvers = backupGR