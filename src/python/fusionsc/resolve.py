"""
Resolution helpers to obtain computable fields and geometries from high-level descriptions

This module maintains a list of active resolvers, which inspect specifications of coils, fields,
and geometries and replace them with concrete specifications that the geometry builder and field
calculators can actively interpret.

The list is maintained in a context variable. Any changes made will be restricted to the active
context (and dependent contexts).
"""

from . import native, data, service, wrappers, warehouse, config
from .asnc import asyncFunction, startEventLoop, wait

from .native.devices import w7x as cppw7x
from .native.devices import jtext as cppjtext
import contextlib
import contextvars

from typing import Union

# Ensure event loop is running
startEventLoop()

_fieldResolvers = contextvars.ContextVar("fusionsc.resolve._fieldResolvers", default = None)
_geometryResolvers = contextvars.ContextVar("fusionsc.resolve._geometryResolvers", default = None)

def _addOfflineResolvers(ref):
	addFieldResolvers([native.offline.fieldResolver(ref)])
	addGeometryResolvers([native.offline.geometryResolver(ref)])

def _addDefaults():
	addFieldResolvers([cppw7x.fieldResolver(), cppjtext.fieldResolver()])
	addGeometryResolvers([cppw7x.geometryResolver(), cppjtext.geometryResolver()])

def reset():
	"""Resets the resolver state to default"""
	_fieldResolvers.set(None)
	_geometryResolvers.set(None)

def fieldResolvers():
	"""Returns the currently active field / coil resolvers"""
	result = _fieldResolvers.get()
	
	if result is not None:
		return result
	
	confCtx = config.context()
	if _fieldResolvers in confCtx:
		return confCtx[_fieldResolvers]
	
	return ()

def geometryResolvers():
	"""Returns the currently active geometry resolvers"""
	result = _geometryResolvers.get()
	
	if result is not None:
		return result
	
	confCtx = config.context()
	if _geometryResolvers in confCtx:
		return confCtx[_geometryResolvers]
	
	return ()

def addFieldResolvers(resolvers):
	"""Adds a resolvers at the end of the currently active list"""
	return _fieldResolvers.set(fieldResolvers() + tuple(resolvers))

def addGeometryResolvers(resolvers):
	"""Adds geometry resolvers at the end of the currently active list"""
	return _geometryResolvers.set(geometryResolvers() + tuple(resolvers))

def connectWarehouse(db):
	"""
	Connects to a warehouse to use for resolution. This will perform the following steps:
	
	- If db is a string, opens the corresponding database (see :py:func:`fusionsc.warehouse.open`)
	- Looks up the entry 'resolveIndex' under the database root and opens it as a DataRef
	  to a service.OfflineData object.
	- Connects resolvers looking up nodes in this object.
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
		
		if isinstance(key, service.MagneticField.ReaderOrBuilder):
			assert isinstance(value, service.MagneticField.ReaderOrBuilder), "Mismatch in key/value types"
			fieldUpdates.append(mapEntry)
		elif isinstance(key, service.Filament.ReaderOrBuilder):
			assert isinstance(value, service.Filament.ReaderOrBuilder), "Mismatch in key/value types"
			coilUpdates.append(mapEntry)
		elif isinstance(key, service.Geometry.ReaderOrBuilder):
			assert isinstance(value, service.Geometry.ReaderOrBuilder), "Mismatch in key/value types"
			geoUpdates.append(mapEntry)
		else:
			assert False, f"Can only use fields, filaments, or geometries in mappings, type(key) = {type(key)}"
	
	return service.OfflineData.newMessage({
		"coils" : coilUpdates,
		"fields" : fieldUpdates,
		"geometries" : geoUpdates
	})	

def updateOfflineData(offlineData, updates):
	"""
	Updates an offline data structure with the data contained in another.
	"""
	assert isinstance(updates, (dict, service.OfflineData.ReaderOrBuilder)), "Updates must be dict or fsc.service.OfflineData"
	
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
 
def importOfflineData(filenameOrData: Union[str, dict, service.OfflineData.ReaderOrBuilder]):
	"""
	Loads the data contained in the given offline archives and uses them to
	perform offline resolution. Alternatively can 
	"""
	if isinstance(filenameOrData, str):
		offlineData = data.openArchive(filenameOrData)
	elif isinstance(filenameOrData, dict):
		offlineData = fsc.data.publish(createOfflineData(filenameOrData))
	else:
		offlineData = fsc.data.publish(filenameOrData)
	
	# Install offline resolvers
	_addOfflineResolvers(offlineData)

@asyncFunction
async def resolveField(field: service.MagneticField.ReaderOrBuilder, followRefs: bool = False) -> service.MagneticField.Reader:	
	"""Processes the field one by one by active resolvers"""
	for r in fieldResolvers():
		field = await r.resolveField(field, followRefs)
		
	return field

@asyncFunction
async def resolveFilament(filament: service.Filament.ReaderOrBuilder, followRefs: bool = False) -> service.Filament.Reader:	
	"""Processes the filament one by one by active resolvers"""	
	for r in fieldResolvers():
		filament = await r.resolveFilament(filament, followRefs)
		
	return filament
	
@asyncFunction
async def resolveGeometry(geometry: service.Geometry.ReaderOrBuilder, followRefs: bool = False) -> service.Geometry.Reader:	
	"""Processes the field one by one by active resolvers"""	
	for r in geometryResolvers():
		geometry = await r.resolveGeometry(geometry, followRefs)
		
	return geometry
