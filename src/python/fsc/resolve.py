from . import native, data
from .asnc import asyncFunction, startEventLoop

from .native.devices import w7x as cppw7x
import contextlib

# Ensure event loop is running
startEventLoop()

# Since there is a small regree of work that the W7-X coil resolver can do even
# without a backing components DB, we add connected to a dummy database.
fieldResolvers = [
	cppw7x.coilsDBResolver(cppw7x.CoilsDB.newDisconnected(""))
]

geometryResolvers = []

def importOfflineData(filename: str):
	"""
	Loads the data contained in the given offline archives and uses them to
	perform offline resolution.
	"""
	offlineData = data.openArchive(filename)
	
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