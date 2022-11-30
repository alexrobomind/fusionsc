from . import native
from . import data

from .asnc import asyncFunction

import numpy as np

def asTagValue(x):
	result = native.TagValue.newMessage()
	
	if x is None:
		result.notSet = None
	elif instanceof(x, int) and x >= 0:
		result.uInt64 = x
	elif instanceof(x, str):
		result.text = x
	else:
		raise "Tag values can only be None, unsigned integer or string"
	
	return result

def cuboid(x1, x2, tags = {}):
    # Prepare mesh data
	x1 = np.asarray(x1)
	x2 = np.asarray(x2)
	
	unitCubeVertices = np.asarray([
		[0, 0, 0],
		[1, 0, 0],
		[1, 1, 0],
		[0, 1, 0],
		[0, 0, 1],
		[1, 0, 1],
		[1, 1, 1],
		[0, 1, 1]
	])
	
	cubeVertices = x1 + (x2 - x1) * unitCubeVertices
	
	indices = [
		# xy surfaces
		0, 1, 2, 3,
		4, 5, 6, 7,
		# xz surfaces
		0, 4, 5, 1,
		2, 6, 7, 3,
		# yz surfaces
		0, 3, 7, 4,
		1, 5, 6, 2
	]
	
    # Put data into mesh struct
	mesh = native.Mesh.newMessage()
	mesh.vertices = cubeVertices
	mesh.indices = indices,
	mesh.polyMesh = [0, 4, 8, 12, 16, 20, 24]
	
    # Publish mesh in distributed data system
	meshRef = data.publish(mesh)
	
    # Put mesh reference & tags into geometry
	import fsc
	result = fsc.Geometry()
	result.geometry.mesh = meshRef
	
	outTags = result.geometry.initTags(len(tags))
	for i, name in enumerate(tags):
		outTags[i].name = name
		outTags[i].value = asTagValue(tags[name])
	
	return result
		

def localGeoLib():
	return native.connectSameThread().newGeometryLib().service

def planarCut(*args, **kwargs):
	return planarCutAsync(*args, **kwargs).wait()

def asPyvista(*args, **kwargs):
	return asPyvistaAsync(*args, **kwargs).wait()

@asyncFunction
async def planarCutAsync(geometry, phi = None, normal = None, center = None):
	assert phi is not None or normal is not None
	assert phi is None or normal is None
	
	geometry = await geometry.resolve.asnc()
	
	request = native.GeometryLib.methods.planarCut.Params.newMessage()
	request.geometry = geometry.geometry
	
	plane = request.plane
	if phi is not None:
		plane.orientation.phi = phi
	if normal is not None:
		plane.orientation.normal = normal
	
	if center is not None:
		plane.center = center
	
	response = await localGeoLib().planarCut(request)
	return np.asarray(response.edges).transpose([2, 0, 1])

def plotCut(geometry, phi = 0, ax = None, plot = True, **kwArgs):
	import matplotlib.pyplot as plt
	import matplotlib
	
	x, y, z = planarCut(geometry, phi = phi)
	r = np.sqrt(x**2 + y**2)
	
	linedata = np.stack([r, z], axis = -1)
	
	coll = matplotlib.collections.LineCollection(linedata, **kwArgs)
	
	if plot:
		if ax is None:
			ax = plt.gca()
		
		ax.add_collection(coll)
	
	return coll

@asyncFunction
async def asPyvistaAsync(geometry):
	import numpy as np
	import pyvista as pv
	
	geometry = await geometry.resolve.asnc()
	
	geoLib = localGeoLib()
	mergedRef = geoLib.merge(geometry.geometry).ref
	mergedGeometry = await data.download.asnc(mergedRef)
	
	def extractMesh(entry):
		mesh = entry.mesh
		
		vertexArray = np.asarray(mesh.vertices)
		indexArray	= np.asarray(mesh.indices)
		
		faces = []
		
		if mesh.which() == 'polyMesh':
			polyRanges = mesh.polyMesh
			
			for iPoly in range(len(polyRanges) - 1):
				start = polyRanges[iPoly]
				end	  = polyRanges[iPoly + 1]
				
				faces.append(end - start)
				faces.extend(indexArray[start:end])
		elif mesh.which() == 'triMesh':
			for offset in range(0, len(indexArray), 3):
				start = offset
				end	  = start + 3
				
				faces.append(end - start)
				faces.extend(indexArray[start:end])
		
		mesh = pv.PolyData(vertexArray, faces)
		return mesh
	
	def extractTags(entry):
		return {
			str(name) : entry.tags[iTag]
			for iTag, name in enumerate(mergedGeometry.tagNames)
		}
	
	def extractEntry(entry):
		mesh = extractMesh(entry)
		
		for tagName, tagValue in extractTags(entry).items():
			if tagValue.which() == 'text':
				mesh.field_data[tagName] = [tagValue.text]
			if tagValue.which() == 'uInt64':
				mesh.field_data[tagName] = [tagValue.uInt64]
				
		return mesh
	
	dataSets = [
		extractEntry(entry)
		for entry in mergedGeometry.entries
	]
	
	return pv.MultiBlock(dataSets)