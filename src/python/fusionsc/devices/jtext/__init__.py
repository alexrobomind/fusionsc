"""J-TEXT parts"""
from ... import service, geometry, magnetics, native

def defaultGrid() -> service.ToroidalGrid.Builder:
	"""
	A 'current best-efford' estimate for a good calculation grid for J-TEXT calculations.
	
	.. note::
	   The default grid might change in the future if a more reasonable tradeoff between
	   accuracy and calculation speed is determined
	"""
	return service.devices.jtext.defaultGrid.value.clone_()

def defaultGeometryGrid() -> service.CartesianGrid.Builder:
	"""
	A 'current best-efford' estimate for a good geometry indexing grid for J-TEXT calculations.
	
	.. note::
	   The default grid might change in the future if a more reasonable tradeoff between
	   accuracy and calculation speed is determined
	"""
	return service.devices.jtext.defaultGeoGrid.value.clone_()

def firstWall() -> geometry.Geometry:
	"""
	J-TEXT plasma vessel geometry
	"""
	result = geometry.Geometry()
	result.data.initJtext().firstWall = None
	return result.withTags({'name': 'First Wall'})

def hfsLimiter() -> geometry.Geometry:
	"""
	High-Field Side Limiter geometry
	"""
	result = geometry.Geometry()
	result.data.initJtext().hfsLimiter = None
	return result.withTags({'name': 'HFS Limiter'})

def target() -> geometry.Geometry:
	"""
	High-Field Side Target geometry
	"""
	result = geometry.Geometry()
	result.data.initJtext().target = None
	return result.withTags({'name': 'HFS Target'})

def lfsLimiter(pos: float) -> geometry.Geometry:
	"""
	Low-field side limiter
	
	:param pos Distance of the limiter to the magnetic axis
	"""
	result = geometry.Geometry()
	result.data.initJtext().lfsLimiter = pos
	return result.withTags({'name': 'LFS Limiter'})

def topLimiter(pos: float) -> geometry.Geometry:
	"""
	Top limiter
	
	:param pos Distance of the limiter to the magnetic axis
	"""
	result = geometry.Geometry()
	result.data.initJtext().topLimiter = pos
	return result.withTags({'name': 'Top Limiter'})

def bottomLimiter(pos: float) -> geometry.Geometry:
	"""
	Bottom limiter
	
	:param pos: Distance of the limiter to the magnetic axis
	"""
	result = geometry.Geometry()
	result.data.initJtext().bottomLimiter = pos
	return result.withTags({'name': 'Bottom Limiter'})

def singleIslandCoil(i: int, current: float) -> magnetics.MagneticConfig:
	"""
	One of the J-TEXT island coils
	
	:param i: Number (1 to 6) of island coil
	:param current: Current to apply to island coil
	"""
	result = magnetics.MagneticConfig()
	
	filField = result.data.initFilamentField()
	
	filField.biotSavartSettings.width = 0.01
	filField.biotSavartSettings.stepSize = 0.01
	
	filField.filament.initJtext().islandCoil = i
	filField.current = current
	
	return result

def islandCoils(currents: list) -> magnetics.MagneticConfig:
	"""
	All 6 island coils with individual currents
	
	:param currents: List of 6 coil current values
	"""
	result = magnetics.MagneticConfig()
	
	result.data.sum = [
		singleIslandCoil(i, currents[i-1]).data
		for i in [1, 2, 3, 4, 5, 6]
	]
	
	return result

def exampleGeqdsk() -> str:
	"""
	An example EFIT GEQDSK file for J-TEXT that can be parsed
	with the EFIT module
	"""
	return native.devices.jtext.exampleGeqdsk()
			
			
def pfcs(limiterPos: float) -> geometry.Geometry:
	"""
	A combination of first wall, limiters, and target geometry.
	
	:param limiterPos: Position of the 3 limiters
	"""
	return firstWall() + hfsLimiter() + lfsLimiter(limiterPos) + bottomLimiter(limiterPos) + topLimiter(limiterPos) + target()