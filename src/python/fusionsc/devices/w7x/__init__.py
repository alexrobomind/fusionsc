"""W7-X parts and IPP site helpers"""

from ... import service, resolve, wrappers

from ...magnetics import MagneticConfig, CoilFilament
from ...geometry import Geometry

from ...native.devices import w7x as w7xnative

from ...data import asyncFunction

from typing import Union

# Databases

def connectCoilsDB(address: str):
	"""
	Connect to the coilsDB webservice at given address and use it to resolve
	W7-X coil specifications
	"""
	coilsDB = w7xnative.webserviceCoilsDB(address)
	
	resolve.fieldResolvers.append(w7xnative.configDBResolver(coilsDB))
	resolve.fieldResolvers.append(w7xnative.coilsDBResolver(coilsDB))
	
	return coilsDB

def connectComponentsDB(address: str):
	"""
	Connect to the componentsDB webservice at given address and use it to resolve
	W7-X geometry specifications
	"""	
	componentsDB = w7xnative.webserviceComponentsDB(address)
	
	resolve.geometryResolvers.append(w7xnative.componentsDBResolver(componentsDB))
	
	return componentsDB

def connectIPPSite():
	"""Connects the resolve module to standard IPP coils DB and components DB"""
	connectCoilsDB("http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest")
	connectComponentsDB("http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest")

class CoilPack(wrappers.structWrapper(service.W7XCoilSet)):
	"""Set of coils that can be used to obtain W7-X specific configurations"""
	pass

@asyncFunction
async def computeCoilFields(calculator, coils: CoilPack, grid = None) -> CoilPack:
	"""Pre-computes the a W7-X coil set on the given grid to be used for different configurations"""
	if grid is None:
		grid = defaultGrid
	
	result = service.W7XCoilSet.newMessage()
	w7xnative.buildCoilFields(coils.data, result.initFields())
	
	async def resolveAndCompute(x):
		x = await resolve.resolveField.asnc(x)
		
		# Computed field is two part: Grid and data ref
		# To hide latency, we extract the reference directly
		
		compField = service.ComputedField.newMessage()
		compField.grid = grid
		compField.data = calculator.compute(x, grid).computedField.data
		
		return compField
	
	fields = result.fields
	
	for i in range(7):
		fields.mainFields[i].computedField = await resolveAndCompute(fields.mainFields[i])
	
	for i in range(5):
		fields.trimFields[i].computedField = await resolveAndCompute(fields.trimFields[i])
	
	for i in range(10):
		fields.controlFields[i].computedField = await resolveAndCompute(fields.controlFields[i])
	
	return CoilPack(result)

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

def coilsDBConfig(id: int) -> MagneticConfig:
	result = fusionsc.magnetics.MagneticConfig()
	result.data.initW7x().configurationDb = id
	
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
	return divertor() + baffles() + heatShield() + pumpSlits() + steelPanels() + vessel() + toroidalClosure()

def op21Geometry():
	return divertor('OP21') + baffles('OP21') + heatShield('OP21') + pumpSlits() + steelPanels() + vessel() + toroidalClosure()
	
# The W7XCoilSet type defaults to the W7-X coils 160 ... 230
defaultCoils = cadCoils('archive')

defaultGrid = w7xnative.defaultGrid()
defaultGeometryGrid = w7xnative.defaultGeometryGrid()