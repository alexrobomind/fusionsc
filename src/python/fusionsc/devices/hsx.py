from .. import service

from ..geometry import Geometry
from ..magnetics import CoilFilament
from .. import service

def defaultGrid():
	grid = service.ToroidalGrid.newMessage()
	grid.rMin = 0.5
	grid.rMax = 2.0
	grid.zMin = -0.5
	grid.zMax = 0.5
	grid.nSym = 4
	grid.nPhi = 32
	grid.nR = 75
	grid.nZ = 50
	
	return grid

def defaultGeometryGrid():
	geoGrid = service.CartesianGrid.newMessage()
	geoGrid.xMin = -1.8
	geoGrid.xMax = 1.8
	geoGrid.yMin = -1.8
	geoGrid.yMax = 1.8
	geoGrid.zMin = -0.5
	geoGrid.zMax = 0.5
	geoGrid.nX = 100
	geoGrid.nY = 100
	geoGrid.nZ = 100
	
	return geoGrid

def wall():
	return Geometry("hsx: firstWall")

def mainCoil(i):
	return CoilFilament({"hsx" : {"mainCoil" : i}})

def auxCoil(i):
	return CoilFilament({"hsx" : {"auxCoil" : i}})

def mainField(currents):
	assert len(currents) == 6
	
	return sum([
		mainCoil(i).biotSavart(current = current)
		for i, current in zip([1, 2, 3, 4, 5, 6], currents)
	])