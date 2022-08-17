from . import native
from . import data

from .asnc import asyncFunction

import numpy as np

def localGeoLib():
	return native.connectSameThread().newGeometryLib().service

def planarCut(*args, **kwargs):
	return planarCutAsync(*args, **kwargs).wait()

def asPyvista(*args, **kwargs):
	return asPyvistaAsync(*args, **kwargs).wait()

@asyncFunction
async def planarCut(geometry, phi = None, normal = None, center = None):
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
async def asPyvista(geometry):
	import numpy as np
	import pyvista as pv
	
	geometry = await geometry.resolve.asnc()
	
	geoLib = localGeoLib()
	mergedRef = geoLib.merge(geometry.geometry).ref
	mergedGeometry = await data.download.asnc(mergedRef)
	
	def extractMesh(entry):
		mesh = entry.mesh
		
		vertexArray = np.asarray(mesh.vertices)
		indexArray  = np.asarray(mesh.indices)
		
		faces = []
		
		if mesh.which() == 'polyMesh':
			polyRanges = mesh.polyMesh
			
			for iPoly in range(len(polyRanges) - 1):
				start = polyRanges[iPoly]
				end   = polyRanges[iPoly + 1]
				
				faces.append(end - start)
				faces.extend(indexArray[start:end])
		elif mesh.which() == 'triMesh':
			for offset in range(0, len(indexArray), 3):
				start = offset
				end   = start + 3
				
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