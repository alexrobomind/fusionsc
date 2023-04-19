from ... import service, geometry, flt

from . import resources

import numpy as np
import importlib.resources as pkg_resources

def defaultGrid(n = 128, nPhi = 128):
	grid = service.ToroidalGrid.newMessage()
	
	grid.nPhi = nPhi
	grid.nR = n
	grid.nZ = n
	grid.rMin = 0.7
	grid.rMax = 1.4
	grid.zMin = -0.35
	grid.zMax = 0.35
	
	return grid

def firstWall(nPhi = 100):
	return geometry.Geometry.from2D(
		r = [0.7135, 1.311, 1.409,  1.409,  1.311,  0.7135, 0.7135],
		z = [0.326 , 0.326, 0.268, -0.268, -0.326, -0.326 , 0.326],
		nPhi = nPhi,
		tags = {"component" : "firstWall"}
	)

def hfsLimiter(nPhi = 100):
	return geometry.Geometry.from2D(
		r = [0.7135, 0.754, 0.754, 0.7135, 0.7135],
		z = [-0.32, -0.32, 0.32, 0.32, -0.32],
		nPhi = nPhi,
		tags = {"component" : "hfsLimiter"}
	)

def lfsLimiter(pos, nPhi = 5):
	phi = np.radians(337.5)
	dr = np.asarray([np.cos(phi), np.sin(phi), 0])
	
	return geometry.Geometry.from2D(
		r = [1.285, 1.315, 1.315, 1.285, 1.285],
		z = [-0.125, -0.125, 0.125, 0.125, -0.125],
		nPhi = nPhi,
		phi1 = np.radians(335.5), phi2 = np.radians(339.5),
		tags = {"component" : "lfsLimiter"}
	).translate(dr * (pos - 0.235))

def topLimiter(pos, nPhi = 5):
	return geometry.Geometry.from2D(
		r = [0.925, 1.175, 1.175, 0.925, 0.925],
		z = [0.235, 0.235, 0.265, 0.265, 0.235],
		nPhi = nPhi,
		phi1 = np.radians(335.5), phi2 = np.radians(339.5),
		tags = {"component" : "topLimiter"}
	).translate([0, 0, pos - 0.235])

def bottomLimiter(pos, nPhi = 5):
	return geometry.Geometry.from2D(
		r = [0.925, 1.175, 1.175, 0.925, 0.925],
		z = [-0.235, -0.235, -0.265, -0.265, -0.235],
		nPhi = nPhi,
		phi1 = np.radians(335.5), phi2 = np.radians(339.5),
		tags = {"component" : "bottomLimiter"}
	).translate([0, 0, 0.235 - pos])

def target(nPhi = 19):
	strData = pkg_resources.read_text(resources, "target.dat")

	r, z = np.asarray([
		line.split("\t")
		for line in strData.split("\n")
	], dtype = np.float64).T

	return geometry.Geometry.from2D(
		r, z,
		nPhi = nPhi,
		phi1 = np.radians(125), phi2 = np.radians(145),
		tags = {"component" : "target"}
	)

def singleIslandCoil(i, current):
	import csv
	csv.register_dialect('skip_space', skipinitialspace=True)
	
	points = []
	
	with pkg_resources.open_text(resources, "GP{}_sco.dat".format(i)) as f:
		r = csv.reader(f, dialect = 'skip_space', delimiter = ' ')
		
		for row in r:
			x, y, z, _ = row
			points.append([x, y, z])
	
	points = np.asarray(points, dtype = np.float64)
	
	result = flt.MagneticConfig()
	filField = result.field.initFilamentField()
	filField.current = current
	filField.biotSavartSettings.width = 0.1
	filField.biotSavartSettings.stepSize = 0.1
	
	filField.filament.inline = points
	
	return result

def islandCoils(currents):
	result = flt.MagneticConfig()
	
	result.field.sum = [
		singleIslandCoil(i, currents[i-1]).field
		for i in [1, 2, 3, 4, 5, 6]
	]
	
	return result
			
			
def pfcs(limiterPos):
	return firstWall() + hfsLimiter() + lfsLimiter(limiterPos) + bottomLimiter(limiterPos) + topLimiter(limiterPos) + target()