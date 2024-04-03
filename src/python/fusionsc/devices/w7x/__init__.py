"""W7-X parts and IPP site helpers"""

from ... import service, resolve, wrappers, flt, backends, remote, warehouse

from ...magnetics import MagneticConfig, CoilFilament
from ...geometry import Geometry

from ...native.devices import w7x as w7xnative

from ...asnc import asyncFunction
from ...wrappers import unstableApi

from ...backends import localResources

from typing import Union, Optional
import numpy as np

# Databases

def _provider():
	return localResources().w7xProvider().pipeline.service

def connectCoilsDB(address: str):
	"""
	Connect to the coilsDB webservice at given address and use it to resolve
	W7-X coil specifications
	"""
	coilsDB = _provider().connectCoilsDb(address).pipeline.service
	
	resolve.fieldResolvers.append(w7xnative.configDBResolver(coilsDB))
	resolve.fieldResolvers.append(w7xnative.coilsDBResolver(coilsDB))
	
	return coilsDB

def connectComponentsDB(address: str):
	"""
	Connect to the componentsDB webservice at given address and use it to resolve
	W7-X geometry specifications
	"""
	componentsDB = _provider().connectComponentsDb(address).pipeline.service
	
	resolve.geometryResolvers.append(w7xnative.componentsDBResolver(componentsDB))
	
	return componentsDB

@unstableApi
def connectLegacyIPPSite():
	"""Connects the resolve module to standard IPP coils DB and components DB"""
	connectCoilsDB("http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest")
	connectComponentsDB("http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest")

def connectIppSite():
	"""Connects the resolve module to the newer fsc-driven Coils- and ComponentsDb proxies"""
	import warnings
	warnings.warn("IPP site backend is experimental, URL might change")
	
	# Set up the W7-X load balancer as main backend
	newBackend = remote.connect("http://sv-coda-wsvc-31:8888/load-balancer")
	backends.alwaysUseBackend(newBackend)
	
	# Load w7xdb (exposed by remote backend) and connect its data index
	resolve.connectWarehouse("remote:w7xdb")
	
	# Note: This is equivalent to
	# database = warehouse.open("remote:w7xdb")
	#  or
	# database = warehouse.openRemote("w7xdb")
	#  followed by
	# resolve.connectWarehouse(database)

class CoilPack(wrappers.structWrapper(service.W7XCoilSet)):
	"""Set of coils that can be used to obtain W7-X specific configurations"""
	
	@asyncFunction
	async def computeFields(self, grid) -> "CoilPack":
		async def resolveAndCompute(x):
			config = MagneticConfig(x)
			config = await config.resolve.asnc()
			config = config.compute(grid)
			
			return config.data.computedField
			
		result = CoilPack()
		w7xnative.buildCoilFields(self.data, result.data.initFields())
		
		fields = result.data.fields
		
		for i in range(7):
			fields.mainFields[i].computedField = await resolveAndCompute(fields.mainFields[i])
		
		for i in range(5):
			fields.trimFields[i].computedField = await resolveAndCompute(fields.trimFields[i])
		
		for i in range(10):
			fields.controlFields[i].computedField = await resolveAndCompute(fields.controlFields[i])
		
		return result
	
	def __await__(self):
		assert self.data.which_() == 'fields', 'Can only await a coil pack on which computeFields() was called'
		
		fields = self.data.fields
		
		async def coro():
			for i in range(7):
				await MagneticConfig(fields.mainFields[i])
			
			for i in range(5):
				await MagneticConfig(fields.trimFields[i])
			
			for i in range(10):
				await MagneticConfig(fields.controlFields[i])
		
		return coro().__await__()

def cadCoils(convention = '1-AA-R0004.5') -> CoilPack:
	"""
	Returns the coil pack for the standard W7-X CAD coils. The winding direction
	of the main coils is adjusted to the requested convention.
	"""
	convention = processCoilConvention(convention)
	
	# The default W7X coil set is the CAD coils
	coilPack = service.W7XCoilSet.newMessage()
	
	if convention == '1-AA-R0004.5':
		coilPack.coils.invertMainCoils = True
	else:
		coilPack.coils.invertMainCoils = False
	
	return CoilPack(coilPack)

def mainField(i_12345 = [15000, 15000, 15000, 15000, 15000], i_ab = [0, 0], coils = None) -> MagneticConfig:
	if coils is None:
		coils = defaultCoils
	
	config = MagneticConfig()
	
	cnc = config.data.initW7x().initCoilsAndCurrents()
	cnc.coils = coils.data
	
	cnc.nonplanar = i_12345
	cnc.planar = i_ab
	
	return config

def trimCoils(i_trim = [0, 0, 0, 0, 0], coils = None) -> MagneticConfig:
	if coils is None:
		coils = defaultCoils
	
	config = MagneticConfig()
	
	cnc = config.data.initW7x().initCoilsAndCurrents()
	cnc.coils = coils.data
	
	cnc.trim = i_trim
	
	return config

def controlCoils(i_cc = [0, 0], coils = None) -> MagneticConfig:
	if coils is None:
		coils = defaultCoils
	
	config = MagneticConfig()
	
	cnc = config.data.initW7x().initCoilsAndCurrents()
	cnc.coils = coils.data
	
	cnc.control = i_cc
	
	return config

def standard(bAx = 2.52, coils = None) -> MagneticConfig:
	return mainField([15000 * bAx / 2.52] * 5, [0] * 2, coils = coils)

def highMirror(bAx = 2.72, coils = None) -> MagneticConfig:
	a = bAx / 2.3
	return mainField([13000 * a, 13260 * a, 14040 * a, 12090 * a, 10959 * a], [0] * 2, coils = coils)

def highIota(bAx = 2.72, coils = None) -> MagneticConfig:
	return mainField([14814.81 * bAx / 2.43] * 5, [-10222.22 * bAx / 2.43] * 2, coils = coils)

def lowIota(bAx = 2.72, coils = None) -> MagneticConfig:
	return mainField([12222.22 * bAx / 2.45] * 5, [9166.67 * bAx / 2.45] * 2, coils = coils)
	

coil_conventions = ['coilsdb', '1-AA-R0004.5', 'archive']
def processCoilConvention(convention):
	assert convention in coil_conventions,	'Invalid coil convention {}, must be one of {}'.format(convention, coil_conventions)
	
	if convention == 'archive':
		return '1-AA-R0004.5'
	
	return convention

def coilsDBConfig(id: int, biotSavartSettings: Optional[service.BiotSavartSettings.ReaderOrBuilder] = None) -> MagneticConfig:
	result = MagneticConfig()
	cdb = result.data.initW7x().initConfigurationDb()
	cdb.configId = id
	cdb.biotSavartSettings = biotSavartSettings
	
	return result

def coilsDBCoil(id: int) -> CoilFilament:
	result = CoilFilament()
	result.data.initW7x().coilsDb = id
	
	return result
	
def component(id) -> Geometry:
	result = Geometry()
	result.data.initW7x().componentsDbMesh = id
	return result

def components(ids = [], name = None) -> Geometry:
	result = sum([component(id) for id in ids])
	
	if name:
		result = result.withTags({'name' : name})
		
	return result
	
def assembly(id) -> Geometry:
	result = Geometry()
	result.data.initW7x().componentsDbAssembly = id
	
	return result

def assemblies(ids = [], name = None) -> Geometry:
	result = sum([assembly(id) for id in ids])
	
	if name:
		result = result.withTags({'name' : name})
		
	return result

def divertor(campaign = 'OP21') -> Geometry:
	if campaign == 'OP12':
		return components(range(165, 170), 'Divertor TDU')
	
	if campaign == 'OP21':
		result = Geometry()
		result.data.initW7x().op21Divertor = None
		return result.withTags({'name' : 'OP2.1 Divertor'})
	
	raise "Unknown campaign " + campaign

def baffles(campaign = 'OP21') -> Geometry:
	if campaign == 'OP12':
		return components(range(320, 325), 'OP1.2 Baffles') + components(range(325, 330), 'OP1.2 Baffle Covers')
	
	if campaign == 'OP21':
		result = Geometry()
		result.data.initW7x().op21Baffles = None
		return result.withTags({'name' : 'OP2.1 Baffles'})
	
	raise "Unknown campaign " + campaign

def heatShield(campaign = 'OP21') -> Geometry:
	if campaign == 'OP12':
		return components(range(330, 335), 'OP1.2 Heat Shield')
	
	if campaign == 'OP21':
		result = Geometry()
		result.data.initW7x().op21HeatShield = None
		return result.withTags({'name' : 'OP2.1 Heat Shield'})
	
	raise "Unknown campaign " + campaign

def pumpSlits():
	return components(range(450, 455), 'Pump Slits')

def steelPanels():
	return assemblies([8], 'Steel Panels')
	
def vessel():
	return assemblies([9], 'Plasma Vessel')

def toroidalClosure():
	return components(range(325, 330), 'Toroidal closure')

def op12Geometry():
	return divertor('OP12') + baffles('OP12') + heatShield('OP12') + pumpSlits() + steelPanels() + vessel() + toroidalClosure()

def op21Geometry():
	return divertor('OP21') + baffles('OP21') + heatShield('OP21') + pumpSlits() + steelPanels() + vessel() + toroidalClosure()
	
# The W7XCoilSet type defaults to the W7-X coils 160 ... 230
defaultCoils = cadCoils('archive')

@unstableApi
def defaultGrid():
	return service.devices.w7x.defaultGrid.value.clone_()

@unstableApi
def defaultGeometryGrid():
	return service.devices.w7x.defaultGeoGrid.value.clone_()

@asyncFunction
def axisCurrent(field, current, grid = None, startPoint = [6.0, 0, 0], stepSize = 0.001, nTurns = 10, nIterations = 10, nPhi = 200, direction = "cw", mapping = None):
	"""
	Variant of fsc.flt.axisCurrent with more reasonable W7-X-tailored default values.
	"""
	if field.data.which_() != 'computedField' and grid is None:
		grid = defaultGrid()
		
	return flt.axisCurrent.asnc(field, current, grid, startPoint, stepSize, nTurns, nIterations, nPhi, direction, mapping)

@asyncFunction
def computeMapping(
	field,
	mappingPlanes = np.radians(72 * np.arange(0, 5)),
	r = np.linspace(4, 7, 200), z = np.linspace(-1.5, 1.5, 200),
	grid = None,
	distanceLimit = 7 * 2 * np.pi / 8,
	padding = 2, numPlanes = 20,
	stepSize = 0.01,
	u0 = [0.5], v0 = [0.5]
):
	"""
	Variant of fsc.flt.computeMapping with more reasonable W7-X-tailored default values.
	"""
	if field.data.which_() != 'computedField' and grid is None:
		grid = defaultGrid()
		
	return flt.computeMapping.asnc(
		field,
		mappingPlanes,
		r, z, grid,
		distanceLimit, padding, numPlanes,
		stepSize, u0, v0
	)
