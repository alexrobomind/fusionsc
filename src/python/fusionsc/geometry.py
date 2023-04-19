"""Geometry processing
"""

from . import data
from . import resolve
from . import service
from . import inProcess

from .asnc import asyncFunction

import numpy as np

class Geometry:
	"""A pythonic wrapper around fusion.service.Geometry.Builder"""
	def __init__(self, geo = None):
		self._holder = service.Geometry.newMessage()
		
		if geo is None:
			self._holder.initNested()
		else:
			self._holder.nested = geo
	
	@property
	def geometry(self):
		return self._holder.nested
	
	@geometry.setter
	def geometry(self, newVal):
		self._holder.nested = newVal
	
	def ptree(self):
		import printree
		printree.ptree(self.geometry)
	
	def graph(self, **kwargs):
		return capnp.visualize(self.geometry, **kwargs)
		
	def __repr__(self):
		return str(self.geometry)
		
	@asyncFunction
	async def resolve(self):
		return Geometry(await resolve.resolveGeometry.asnc(self.geometry))
	
	def __add__(self, other):
		if not isinstance(other, Geometry):
			return NotImplemented
			
		result = Geometry()
		
		if self.geometry.which() == 'combined' and other.geometry.which() == 'combined':
			result.geometry.combined = list(self.geometry.combined) + list(other.geometry.combined)
			return result
		
		if self.geometry.which() == 'combined':
			result.geometry.combined = list(self.geometry.combined) + [other.geometry]
			return result
		
		if other.geometry.which() == 'combined':
			result.geometry.combined = [self.geometry] + list(other.geometry.combined)
			return result
		
		result.geometry.combined = [self.geometry, other.geometry]
		return result
	
	def translate(self, dx):
		result = Geometry()
		
		transformed = result.geometry.initTransformed()
		shifted = transformed.initShifted()
		shifted.shift = dx
		shifted.node.leaf = self.geometry
		
		return result
	
	def rotate(self, angle, axis, center = [0, 0, 0]):
		result = Geometry()
		
		transformed = result.geometry.initTransformed()
		turned = transformed.initTurned()
		turned.axis = axis
		turned.angle = angle
		turned.center = center
		turned.node.leaf = self.geometry
		
		return result
	
	@staticmethod
	def polyMesh(vertices, polyIndices):
		"""
		Creates a polygon mesh from a [N, 3] array-like of vertices and a list of polygins (which is each a list of vertex indices)
		"""
		vertices = np.asarray(vertices)
		assert len(vertices.shape) == 2
		assert vertices.shape[1] == 3
		
		meshIndices = np.concatenate(polygons)
		
		polyBoundaries = [0]
		offset = 0
		isTrimesh = True
		
		for poly in polygons:
			offset += len(poly)
			polyBoundaries.append(offset)
			
			if len(poly) != 3:
				isTrimesh = False
		
		
		mesh = service.Mesh.initMessage()
		mesh.vertices = vertices
		mesh.indices = meshIndices
		
		if isTrimesh:
			mesh.triMesh = None
		else:
			mesh.polyMesh = polyBoundaries
		
		meshRef = data.publish(mesh)
		del mesh
		
		result = Geometry()
		result.geometry.mesh = meshRef
		
		return result
	
	@asyncFunction
	@staticmethod
	async def load(filename):
		geo = await data.readArchive.asnc(filename)
		return Geometry(geo)
	
	@staticmethod
	def from2D(r, z, nPhi = 100, phi1 = None, phi2 = None, tags = {}, close = True):
		result = Geometry()
		
		wt = result.geometry.initWrapToroidally()
		wt.nPhi = nPhi
		wt.r = r
		wt.z = z
		
		if phi1 is not None:
			range = wt.initPhiRange()
			range.phiStart = phi1
			range.phiEnd = phi2
			range.close = close
	
		outTags = result.geometry.initTags(len(tags))
		for i, name in enumerate(tags):
			outTags[i].name = name
			outTags[i].value = asTagValue(tags[name])
		
		return result
	
	@asyncFunction
	async def save(self, filename):
		await data.writeArchive.asnc(self.geometry, filename)

def asTagValue(x):
	"""Convert a possible tag value into an instance of fsc.service.TagValue"""
	result = service.TagValue.newMessage()
	
	if x is None:
		result.notSet = None
	elif isinstance(x, int) and x >= 0:
		result.uInt64 = x
	elif isinstance(x, str):
		result.text = x
	else:
		raise "Tag values can only be None, unsigned integer or string"
	
	return result

def cuboid(x1, x2, tags = {}):
	"""Creates a cuboid between x1 and x2"""
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
	mesh = service.Mesh.newMessage()
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
	"""Creates an in-thread GeometryLib instance"""
	return inProcess.root().newGeometryLib().service

@asyncFunction
async def planarCut(geometry, phi = None, normal = None, center = None):
	"""Computes a planar cut of the geometry along either a given plane or a given phi plane"""
	assert phi is not None or normal is not None
	assert phi is None or normal is None
	
	geometry = await geometry.resolve.asnc()
	
	request = service.GeometryLib.methods.planarCut.Params.newMessage()
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
	"""Plot the phi cut of a given geometry in either the given axes or the current active matplotlib axes"""
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
	"""Convert the given geometry into a PyVista / VTK mesh"""
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