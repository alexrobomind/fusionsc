from ... import service, geometry, flt, native

def defaultGrid() -> service.ToroidalGrid.Reader:
	return native.devices.jtext.defaultGrid()

def defaultGeometryGrid() -> service.CartesianGrid.Reader:
	return native.devices.jtext.defaultGeometryGrid()

def firstWall():
	result = geometry.Geometry()
	result.geometry.initJtext().firstWall = None
	return result

def hfsLimiter():
	result = geometry.Geometry()
	result.geometry.initJtext().hfsLimiter = None
	return result

def target():
	result = geometry.Geometry()
	result.geometry.initJtext().target = None
	return result

def lfsLimiter(pos):
	result = geometry.Geometry()
	result.geometry.initJtext().lfsLimiter = pos
	return result

def topLimiter(pos):
	result = geometry.Geometry()
	result.geometry.initJtext().topLimiter = pos
	return result

def bottomLimiter(pos):
	result = geometry.Geometry()
	result.geometry.initJtext().bottomLimiter = pos
	return result

def singleIslandCoil(i, current):
	result = flt.MagneticConfig()
	
	filField = result.field.initFilamentField()
	
	filField.biotSavartSettings.width = 0.01
	filField.biotSavartSettings.stepSize = 0.01
	
	filField.filament.initJtext().islandCoil = i
	
	return result

def islandCoils(currents):
	result = flt.MagneticConfig()
	
	result.field.sum = [
		singleIslandCoil(i, currents[i-1]).field
		for i in [1, 2, 3, 4, 5, 6]
	]
	
	return result

def exampleGeqdsk() -> str:
	return native.devices.jtext.exampleGeqdsk()
			
			
def pfcs(limiterPos):
	return firstWall() + hfsLimiter() + lfsLimiter(limiterPos) + bottomLimiter(limiterPos) + topLimiter(limiterPos) + target()