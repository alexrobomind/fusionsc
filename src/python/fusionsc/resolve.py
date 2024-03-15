"""Resolution helpers to obtain computable fields and geometries from high-level descriptions"""

from . import native, data, service, wrappers, warehouse
from .asnc import asyncFunction, startEventLoop, wait

from .native.devices import w7x as cppw7x
from .native.devices import jtext as cppjtext
import contextlib
import contextvars

# Ensure event loop is running
startEventLoop()

_fieldResolvers = contextvars.ContextVar("fusionsc.resolve._fieldResolvers", default = None)
_geometryResolvers = contextvars.ContextVar("fusionsc.resolve._geometryResolvers", default = None)

def _addOfflineResolvers(ref):
	addFieldResolvers([native.offline.fieldResolver(ref)])
	addGeometryResolvers([native.offline.geometryResolver(ref)])

def reset():
	_fieldResolvers.set(None)
	_geometryResolvers.set(None)

def fieldResolvers():
	result = _fieldResolvers.get()
	
	if result is not None:
		return result
		
	return (cppw7x.fieldResolver(), cppjtext.fieldResolver())

def geometryResolvers():
	result = _geometryResolvers.get()
	
	if result is not None:
		return result
		
	return (cppw7x.geometryResolver(), cppjtext.geometryResolver())

def addFieldResolvers(resolvers):
	return _fieldResolvers.set(fieldResolvers() + tuple(resolvers))

def addGeometryResolvers(resolvers):
	return _geometryResolvers.set(geometryResolvers() + tuple(resolvers))

def connectWarehouse(db):
	"""
	Connects to a warehouse to use for resolution
	"""
	# Optionally open database
	if isinstance(db, str):
		db = warehouse.open(db)
		
	# Open root entry
	rootEntry = db.get("resolveIndex")
	ref = rootEntry.ref
	
	# Install offline resolvers
	_addOfflineResolvers(ref)

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
		
		if isinstance(key, (service.MagneticField.Builder, service.MagneticField.Reader)):
			assert isinstance(value, (service.MagneticField.Builder, service.MagneticField.Reader)), "Mismatch in key/value types"
			fieldUpdates.append(mapEntry)
		elif isinstance(key, (service.Filament.Builder, service.Filament.Reader)):
			assert isinstance(value, (service.Filament.Builder, service.Filament.Reader)), "Mismatch in key/value types"
			coilUpdates.append(mapEntry)
		elif isinstance(key, (service.Geometry.Builder, service.Geometry.Reader)):
			assert isinstance(value, (service.Geometry.Builder, service.Geometry.Reader)), "Mismatch in key/value types"
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
		root = await data.download.asnc(ref)
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
	_addOfflineResolvers(offlineData)

@asyncFunction
async def resolveField(field, followRefs: bool = False):		
	for r in fieldResolvers():
		field = await r.resolveField(field, followRefs)
		
	return field

@asyncFunction
async def resolveFilament(filament, followRefs: bool = False):		
	for r in fieldResolvers():
		filament = await r.resolveFilament(filament, followRefs)
		
	return filament
	
@asyncFunction
async def resolveGeometry(geometry, followRefs: bool = False):		
	for r in geometryResolvers():
		geometry = await r.resolveGeometry(geometry, followRefs)
		
	return geometry
