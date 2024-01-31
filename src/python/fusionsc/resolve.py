"""Resolution helpers to obtain computable fields and geometries from high-level descriptions"""

from . import native, data, service, wrappers, warehouse
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

@asyncFunction
async def connectWarehouse(db):
	"""
	Connects to a warehouse to use for resolution
	"""
	# Optionally open database
	if isinstance(db, str):
		db = await warehouse.open.asnc(db)
		
	# Open root entry
	rootEntry = await db.get.asnc("resolveIndex")
	ref = rootEntry.ref
	
	# Install offline resolvers
	fieldResolvers.append(native.offline.fieldResolver(ref))
	geometryResolvers.append(native.offline.geometryResolver(ref))

def createOfflineData(data: dict):
	"""
	Creates an offline data structure from a key-value mapping of geometries,
	fields, and coils.
	"""
	geoUpdates = []
	fieldUpdates = []
	coilUpdates = []
	
	for key, value in data.items():
		if isinstance(key, wrappers.StructWrapperBase):
			key = key.data
		
		if isinstance(value, wrappers.StructWrapperBase):
			value = value.data
		
		mapEntry = {"key" : key, "val" : value}
		
		if isinstance(key, service.MagneticField):
			assert isinstance(value, service.MagneticField), "Mismatch in key/value types"
			fieldUpdates.append(mapEntry)
		elif isinstance(key, service.Filament):
			assert isinstance(value, service.Filament), "Mismatch in key/value types"
			coilUpdates.append(mapEntry)
		elif isinstance(key, service.Geometry):
			assert isinstance(value, service.Geometry), "Mismatch in key/value types"
			geoUpdates.append(mapEntry)
		else:
			assert False, "Can only use fields, filaments, or geometries in mappings"
	
	return service.OfflineData.newMessage({
		"coils" : coilUpdates,
		"fields" : fieldUpdates,
		"geometries" : geoUpdates
	})	

def updateOfflineData(offlineData, updates):
	"""
	Updates an offline data structure with the data contained in another.
	"""
	assert isinstance(updates, (dict, service.OfflineData)), "Updates must be dict or fsc.service.OfflineData"
	
	if isinstance(updates, dict):
		updates = createOfflineData(updates)
	
	native.offline.updateOfflineData(offlineData, updates)

@asyncFunction
async def updateWarehouse(db, updates):
	"""
	Updates the contents of a warehouse to be used for data resolution
	"""
	# Optionally open database
	if isinstance(db, str):
		db = await warehouse.open.asnc(db)
	
	# Open root entry
	if "resolveIndex" in await db.ls.asnc():
		rootEntry = await db.get.asnc("resolveIndex")
		ref = rootEntry.ref
	
		# Download root entry
		root = await fsc.data.download.asnc(ref)
		root = root.clone_()
	else:
		root = service.OfflineData.newMessage()
	
	# Update structure
	updateOfflineData(root, updates)
	
	# Store updated data (only uploads index and missing data)
	await db.put.asnc("resolveIndex", root)
	

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