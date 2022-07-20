from . import native
from .asnc import asyncFunction

from .native.devices import w7x as cppw7x
import contextlib

fieldResolvers = []
geometryResolvers = []

def importOfflineData(filename: str):
	"""
	Loads the data contained in the given offline archives and uses them to
	perform offline resolution.
	"""
	offlineData = native.loadArchive(filename)
	
	# =============== W7-X specifics ==================
	
	# Create offline Coils- and Components-DBs and connect resolvers
	coilsDB = cppw7x.offlineCoilsDB(offlineData)
	coilsDBResolver = cppw7x.coilsDBResolver(coilsDB)
	
	componentsDB = cppw7x.offlineComponentsDB(offlineData)
	componentsDBResolver = cppw7x.componentsDBResolver(componentsDB)

	fieldResolvers.append(coilsDBResolver)
	geometryResolvers.append(componentsDBResolver)

@asyncFunction
async def resolveField(field, followRefs: bool = False):		
	for r in fieldResolvers:
		try:
			field = await r.resolveField(field, followRefs)
		except:
			pass
		
	return field
	
@asyncFunction
async def resolveGeometry(geometry, followRefs: bool = False):		
	for r in geometryResolvers:
		try:
			geometry = await r.resolve(geometry, followRefs)
		except:
			pass
		
	return geometry

@contextlib.contextmanager
def backupResolvers():
	global fieldResolvers
	global geometryResolvers
	
	backupFR = fieldResolvers.copy()
	backupGR = geometryResolvers.copy()
	
	try:
		yield None
	finally:
		fieldResolvers = backupFR
		geometryResolvers = backupGR