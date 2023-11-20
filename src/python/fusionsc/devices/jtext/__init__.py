"""J-TEXT parts"""
from ... import service, geometry, magnetics, native

def defaultGrid() -> service.ToroidalGrid.Reader:
	return service.devices.jtext.defaultGrid.value.clone_()

def defaultGeometryGrid() -> service.CartesianGrid.Reader:
	return service.devices.jtext.defaultGeoGrid.value.clone_()

def firstWall():
	result = geometry.Geometry()
	result.data.initJtext().firstWall = None
	return result.withTags({'name': 'First Wall'})

def hfsLimiter():
	result = geometry.Geometry()
	result.data.initJtext().hfsLimiter = None
	return result.withTags({'name': 'HFS Limiter'})

def target():
	result = geometry.Geometry()
	result.data.initJtext().target = None
	return result.withTags({'name': 'HFS Target'})

def lfsLimiter(pos):
	result = geometry.Geometry()
	result.data.initJtext().lfsLimiter = pos
	return result.withTags({'name': 'LFS Limiter'})

def topLimiter(pos):
	result = geometry.Geometry()
	result.data.initJtext().topLimiter = pos
	return result.withTags({'name': 'Top Limiter'})

def bottomLimiter(pos):
	result = geometry.Geometry()
	result.data.initJtext().bottomLimiter = pos
	return result.withTags({'name': 'Bottom Limiter'})

def singleIslandCoil(i, current):
	result = magnetics.MagneticConfig()
	
	filField = result.data.initFilamentField()
	
	filField.biotSavartSettings.width = 0.01
	filField.biotSavartSettings.stepSize = 0.01
	
	filField.filament.initJtext().islandCoil = i
	filField.current = current
	
	return result

def islandCoils(currents):
	result = magnetics.MagneticConfig()
	
	result.data.sum = [
		singleIslandCoil(i, currents[i-1]).data
		for i in [1, 2, 3, 4, 5, 6]
	]
	
	return result

def exampleGeqdsk() -> str:
	return native.devices.jtext.exampleGeqdsk()
			
			
def pfcs(limiterPos):
	return firstWall() + hfsLimiter() + lfsLimiter(limiterPos) + bottomLimiter(limiterPos) + topLimiter(limiterPos) + target()