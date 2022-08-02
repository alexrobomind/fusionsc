import fsc
import fsc.native.devices.w7x as w7xnative

from fsc import native, resolve, flt
from fsc.asnc import asyncFunction, eager

from typing import Union

# Databases

def connectCoilsDB(address: str):
	"""
	Connect to the coilsDB webservice at given address and use it to resolve
	W7-X coil specifications
	"""
	coilsDB = w7xnative.webserviceCoilsDB(address)
	coilsDBResolver = w7xnative.coilsDBResolver(coilsDB)
	resolve.fieldResolvers.append(coilsDBResolver)
	return coilsDB

def connectComponentsDB(address: str):
	"""
	Connect to the componentsDB webservice at given address and use it to resolve
	W7-X geometry specifications
	"""	
	componentsDB = w7xnative.webserviceComponentsDB(address)
	componentsDBResolver = w7xnative.componentsDBResolver(componentsDB)
	resolve.geometryResolvers.append(componentsDBResolver)
	return componentsDB

@eager
async def computeCoilFields(calculator, coils: Union[native.W7XCoilSet.Builder, native.W7XCoilSet.Reader], grid = None) -> native.W7XCoilSet.Builder:
	if grid is None:
		grid = defaultGrid
	
	result = native.W7XCoilSet.newMessage()
	w7xnative.buildCoilFields(coils, result.initFields())
	
	async def resolveAndCompute(x):
		x = await resolve.resolveField(x)
		x = (await calculator.compute(x, grid)).computedField
		return x
	
	fields = result.fields
	
	for i in range(7):
		print("Main", i)
		fields.mainFields[i].computedField = await resolveAndCompute(fields.mainFields[i])
	
	for i in range(5):
		print("Trim", i)
		fields.trimFields[i].computedField = await resolveAndCompute(fields.trimFields[i])
	
	for i in range(10):
		print("CC", i)
		fields.controlFields[i].computedField = await resolveAndCompute(fields.controlFields[i])
	
	return result

def cadCoils(convention = '1-AA-R0004.5') -> native.W7XCoilSet:
	"""
	Returns the coil pack for the standard W7-X CAD coils. The winding direction
	of the main coils is adjusted to the requested convention.
	"""
	convention = processCoilConvention(convention)
	
	# The native W7X coil set is the CAD coils
	coilPack = native.W7XCoilSet.newMessage()
	
	if convention == '1-AA-R0004.5':
		coilPack.coils.invertMainCoils = True
	else:
		coilPack.coils.invertMainCoils = False
	
	return coilPack

def mainField(i_12345 = [15000, 15000, 15000, 15000, 15000], i_ab = [0, 0], coils = None) -> "fsc.MagneticConfig":
	if coils is None:
		coils = defaultCoils
	
	config = fsc.MagneticConfig()
	
	cnc = config.field.initW7xMagneticConfig().initCoilsAndCurrents()
	cnc.coils = coils
	
	cnc.nonplanar = i_12345
	cnc.planar = i_ab
	
	return config

def trimCoils(i_trim = [0, 0, 0, 0, 0], coils = None) -> "fsc.MagneticConfig":
	if coils is None:
		coils = defaultCoils
	
	config = fsc.MagneticConfig()
	
	cnc = config.field.initW7xMagneticConfig().initCoilsAndCurrents()
	cnc.coils = coils
	
	cnc.trim = i_trim
	
	return config

def controlCoils(i_cc = [0, 0], coils = None) -> "fsc.MagneticConfig":
	if coils is None:
		coils = defaultCoils
	
	config = fsc.MagneticConfig()
	
	cnc = config.field.initW7xMagneticConfig().initCoilsAndCurrents()
	cnc.coils = coils
	
	cnc.control = i_cc
	
	return config

def standard(bAx = 2.52, coils = None) -> "fsc.MagneticConfig":
	return mainField([15000 * bAx / 2.52] * 5, [0] * 2, coils = coils)

coil_conventions = ['coilsdb', '1-AA-R0004.5', 'archive']
def processCoilConvention(convention):
	assert convention in coil_conventions,	'Invalid coil convention {}, must be one of {}'.format(convention, coil_conventions)
	
	if convention == 'archive':
		return '1-AA-R0004.5'
	
	return convention

def components(ids = []):
	result = fsc.Geometry()
	result.geometry.componentsDBMeshes = ids
	return result

def divertor():
	return components(range(165, 170))
	
# The W7XCoilSet type defaults to the W7-X coils 160 ... 230
defaultCoils = cadCoils('archive')

defaultGrid = w7xnative.defaultGrid()
defaultGeometryGrid = w7xnative.defaultGeometryGrid()