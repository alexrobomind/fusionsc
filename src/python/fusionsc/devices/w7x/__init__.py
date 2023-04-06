import fusionsc
import fusionsc.native.devices.w7x as w7xnative

from fusionsc import service, resolve, flt
from fusionsc.asnc import asyncFunction

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

def connectIPPSite():
	connectCoilsDB("http://esb.ipp-hgw.mpg.de:8280/services/CoilsDBRest")
	connectComponentsDB("http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest")

@asyncFunction
async def computeCoilFields(calculator, coils: Union[service.W7XCoilSet.Builder, service.W7XCoilSet.Reader], grid = None) -> service.W7XCoilSet.Builder:
	if grid is None:
		grid = defaultGrid
	
	result = service.W7XCoilSet.newMessage()
	w7xnative.buildCoilFields(coils, result.initFields())
	
	async def resolveAndCompute(x):
		x = await resolve.resolveField.asnc(x)
		
		#x = (await calculator.compute(x, grid)).computedField
		
		# Cheatycheat cheatie
		# Computed field is two part: Grid and data ref
		# To hide latency, we extract the reference directly
		
		compField = service.ComputedField.newMessage()
		compField.grid = grid
		compField.data = calculator.compute(x, grid).computedField.data
		
		return compField
	
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

def highMirror(bAx = 2.72, coils = None)  -> "fsc.MagneticConfig":
	a = bAx / 2.3
	return mainField([13000 * a, 13260 * a, 14040 * a, 12090 * a, 10959 * a], [0] * 2, coils = coils)

def highIota(bAx = 2.72, coils = None) -> "fsc.MagneticConfig":
	return mainField([14814.81 * field / 2.43] * 5, [-10222.22 * field / 2.43] * 2, coils = coils)

def lowIota(bAx = 2.72, coils = None) -> "fsc.MagneticConfig":
	return mainField([12222.22 * field / 2.45] * 5, [9166.67 * field / 2.45] * 2, coils = coils)

coil_conventions = ['coilsdb', '1-AA-R0004.5', 'archive']
def processCoilConvention(convention):
	assert convention in coil_conventions,	'Invalid coil convention {}, must be one of {}'.format(convention, coil_conventions)
	
	if convention == 'archive':
		return '1-AA-R0004.5'
	
	return convention

def components(ids = [], name = None):
	result = fsc.Geometry()
	result.geometry.componentsDBMeshes = ids
	
	if name:
		tag = result.geometry.initTags(1)[0]
		tag.name = 'name'
		tag.value.text = name
		
	return result

def assemblies(ids = [], name = None):
	result = fsc.Geometry()
	result.geometry.componentsDBAssemblies = ids
	
	if name:
		tag = result.geometry.initTags(1)[0]
		tag.name = 'name'
		tag.value.text = name
		
	return result

def divertor(campaign = 'OP21'):
    if campaign == 'OP12':
        return components(range(165, 170), 'Divertor TDU')
    
    if campaign == 'OP21':
        return fsc.Geometry(w7xnative.op21Divertor())
    
    raise "Unknown campaign " + campaign

def baffles(campaign = 'OP21'):
    if campaign == 'OP12':
        return components(range(320, 325), 'OP1.2 Baffles') + components(range(325, 330), 'OP1.2 Baffle Covers')
    
    if campaign == 'OP21':
        return fsc.Geometry(w7xnative.op21BafflesNoHoles())
    
    raise "Unknown campaign " + campaign

def heatShield(campaign = 'OP21'):
    if campaign == 'OP12':
        return components(range(330, 335), 'OP1.2 Heat Shield')
    
    if campaign == 'OP21':
        return fsc.Geometry(w7xnative.op21HeatShieldNoHoles())
    
    raise "Unknown campaign " + campaign

def pumpSlits():
	return components(range(450, 455), 'Pump Slits')

def steelPanels():
	return assemblies([8], 'Steel Panels')
	
def vessel():
	return assemblies([9], 'Plasma Vessel')

def op12Geometry():
	return divertor() + baffles() + heatShield() + pumpSlits() + steelPanels() + vessel()

def op21Geometry():
    return divertor('OP21') + baffles('OP21') + heatShield('OP21') + pumpSlits() + steelPanels() + vessel()
	
# The W7XCoilSet type defaults to the W7-X coils 160 ... 230
defaultCoils = cadCoils('archive')

defaultGrid = w7xnative.defaultGrid()
defaultGeometryGrid = w7xnative.defaultGeometryGrid()