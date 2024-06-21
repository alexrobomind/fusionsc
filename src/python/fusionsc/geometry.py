"""Geometry processing"""

from . import data
from . import resolve
from . import service
from . import backends
from . import wrappers
from . import native

from .asnc import asyncFunction
from ._api_markers import unstableApi

import numpy as np

class Geometry(wrappers.structWrapper(service.Geometry)):	
	@asyncFunction
	async def resolve(self):
		"""Returns a new geometry with all unresolved nodes replaced by their resolved equivalents."""
		return Geometry(await resolve.resolveGeometry.asnc(self.data))
	
	@asyncFunction
	async def merge(self):
		"""Creates a merged geometry (all meshes combined into one big message)"""
		if self.data.which_() == 'merged':
			return Geometry(self.data)
		
		resolved = await self.resolve.asnc()
		mergedRef = geometryLib().merge(resolved.data).pipeline.ref
		
		result = Geometry()
		result.data.merged = mergedRef
		return result
	
	@asyncFunction
	async def reduce(self, maxVerts = 1000000, maxIndices = 1000000):
		"""Creates a new geometry with small meshes combined into a small number of large meshes"""
		resolved = await self.resolve.asnc()
		
		reducedRef = geometryLib().reduce(resolved.data, maxVerts, maxIndices).pipeline.ref
		
		result = Geometry()
		result.data.merged = reducedRef
		return result
	
	@asyncFunction
	async def getMerged(self):
		"""After merging the geometry, downloads the merged data and returns them"""
		merged = await self.merge.asnc()
		return await data.download.asnc(merged.data.merged)
	
	@asyncFunction
	async def index(self, geometryGrid):
		"""Computes an indexed geometry for accelerated intersection tests"""
		if geometryGrid is None:
			assert self.data.which_() == 'indexed', 'Must specify geometry grid or use pre-indexed geometry'
			return Geometry(self.data)
		
		resolved = await self.resolve.asnc()
		
		result = Geometry()
		indexed = result.data.initIndexed()
		indexed.grid = geometryGrid
		
		indexPipeline = geometryLib().index(resolved.data, geometryGrid).pipeline.indexed
		indexed.base = indexPipeline.base
		indexed.data = indexPipeline.data
		
		return result
	
	def __await__(self):
		assert self.data.which_() == 'indexed' or self.data.which_() == 'merged', (
			'Can only await merged or indexed geometries, but geometry was "' + self.which_() + "'." +
			' Schedule a merge(...) or index(...) computation first.'
		)
		
		if self.data.which_() == 'indexed':
			return self.data.indexed.data.__await__()
		
		if self.data.which_() == 'merged':
			return self.data.merged.__await__()
	
	@asyncFunction
	async def intersect(self, pStart, pEnd, grid = None):
		"""Computes a line intersection test"""
		indexed = await self.index.asnc(grid)
		
		response = await geometryLib().intersect(indexed.data.indexed, pStart, pEnd)
		
		return {
			'lambda' : np.asarray(response['lambda']),
			'position' : np.asarray(response.position),
			'tags' : {
				e.name : np.asarray(e.values)
				for e in response.tags
			}
		}
	
	def __add__(self, other):
		if isinstance(other, int) and other == 0:
			return self
		
		if not isinstance(other, Geometry):
			return NotImplemented
			
		result = Geometry()
		
		# If the geometries have tags, we can not merge the lists either way.
		# In this case, just return a simple sum
		def canAbsorb(x):
			return x.data.which_() == 'combined' and len(x.data.tags) == 0
		
		if canAbsorb(self) and canAbsorb(other):
			result.data.combined = list(self.data.combined) + list(other.data.combined)
			return result
		
		if canAbsorb(self):
			result.data.combined = list(self.data.combined) + [other.data]
			return result
		
		if canAbsorb(other):
			result.data.combined = [self.data] + list(other.data.combined)
			return result
		
		result.data.combined = [self.data, other.data]
		return result
	
	def __radd__(self, other):
		return self.__add__(other)
	
	def translate(self, dx):
		"""Returns a new geometry shifted by the given vector"""
		result = Geometry()
		
		transformed = result.data.initTransformed()
		shifted = transformed.initShifted()
		shifted.shift = dx
		
		if self.data.which_() == "transformed":
			shifted.node = self.data.transformed
		else:
			shifted.node.leaf = self.data
		
		return result
	
	def scale(self, by = 1.0):
		"""Returns a geometry scaled by the given factor"""
		import numbers
		
		if isinstance(by, numbers.Number):
			by = [by, by, by]
		
		result = Geometry()
		transformed = result.data.initTransformed()
		scaled = transformed.initScaled()
		scaled.scale = by
		
		if self.data.which_() == "transformed":
			scaled.node = self.data.transformed
		else:
			scaled.node.leaf = self.data
		
		return result
	
	def rotate(self, angle, axis, center = [0, 0, 0]):
		"""Returns a new geometry rotated around the prescribed axis and center point"""
		result = Geometry()
		
		transformed = result.data.initTransformed()
		turned = transformed.initTurned()
		turned.axis = axis
		turned.angle.rad = angle
		turned.center = center
		
		if self.data.which_() == "transformed":
			turned.node = self.data.transformed
		else:
			turned.node.leaf = self.data
		
		return result
	
	def withTags(self, extraTags):
		"""Returns a new geometry with the given tags"""
		result = Geometry(self.data)
		
		oldTags = { tag.name : tag.value for tag in self.data.tags }
		newKeys = set(list(oldTags) + list(extraTags))
		
		result.data.tags = [
			{
				'name' : name,
				'value' : (
					asTagValue(extraTags[name])
					if name in extraTags
					else oldTags[name]
				)
			}
			for name in newKeys
		]
		
		return result
	
	def filter(self, filter):
		"""Restricts the geometry to meshes where the given tags meet one (or more) specified values"""		
		if not isinstance(filter, service.GeometryFilter.ReaderOrBuilder):
			assert isinstance(filter, dict)
			
			def processEntry(tag, val):
				if not isinstance(val, list):
					val = [val]
				
				val = [asTagValue(e) for e in val]
				
				return {
					"isOneOf" : {
						"tagName" : tag,
						"values" : val
					}
				}
			
			filter = service.GeometryFilter.newMessage({
				"and" : [
					processEntry(k, v) for k, v in filter.items()
				]
			})
		
		return Geometry({"filter" : {"filter" : filter, "geometry" : self.data}})
	
	@asyncFunction
	async def download(self):
		"""For a geometry of 'ref' type, downloads the contents of the referenced geometry and wraps them in a new Geometry instance"""
		if self.data.which_() != 'ref':
			return Geometry(self.data)
		
		ref = self.data.ref
		downloaded = await data.download.asnc(ref)
		
		return Geometry(downloaded)
	
	@staticmethod
	def polyMesh(vertices, polyIndices):
		"""
		Creates a polygon mesh from a [N, 3] array-like of vertices and a list of polygins (which is each a list of vertex indices)
		"""
		vertices = np.asarray(vertices)
		assert len(vertices.shape) == 2
		assert vertices.shape[1] == 3
		
		meshIndices = np.concatenate(polyIndices)
		
		polyBoundaries = [0]
		offset = 0
		isTrimesh = True
		
		for poly in polyIndices:
			offset += len(poly)
			polyBoundaries.append(offset)
			
			if len(poly) != 3:
				isTrimesh = False
		
		
		mesh = service.Mesh.newMessage()
		mesh.vertices = vertices
		mesh.indices = meshIndices
		
		if isTrimesh:
			mesh.triMesh = None
		else:
			mesh.polyMesh = polyBoundaries
		
		meshRef = data.publish(mesh)
		del mesh
		
		result = Geometry()
		result.data.mesh = meshRef
		
		return result
	
	@staticmethod
	def from2D(r, z, nPhi = 100, phi1 = None, phi2 = None, close = True, tags = {}):
		"""Creates a geometry out of a 2D RZ array by wrapping it toroidally"""
		result = Geometry()
		
		wt = result.data.initWrapToroidally()
		wt.nPhi = nPhi
		wt.r = r
		wt.z = z
		
		if phi1 is not None:
			assert phi2 is not None, "Must either specify ph1 and phi2 or neither"
			
			range = wt.initPhiRange()
			range.phiStart.rad = phi1
			range.phiEnd.rad = phi2
			range.close = close
			
		return result.withTags(tags)
	
	@staticmethod
	def quadMesh(vertices, wrapU = False, wrapV = False):
		"""
		Creates a geometry out of a quad mesh
		
		params:
		  - vertices: A numpy array-like of shape [3, nU, nV] containing the individual vertices
		  - wrapU: Whether to link the first and last slice of the "u" dimension
		  - wrapV: Whether to link the first and last slice of the "v" dimension
		"""
		asTensor = service.Float64Tensor.newMessage(np.transpose(vertices, [1, 2, 0]))
		
		return Geometry({
			"quadMesh" : {
				"vertices" : data.publish(asTensor),
				"wrapU" : wrapU,
				"wrapV" : wrapV
			}
		})
	
	@staticmethod
	@unstableApi
	def fromPyvista(polyData):
		"""Creates a geometry from a pyvista.PolyData object"""
		vertices = polyData.verts
		
		linesIn = polyData.lines
		linesOut = []
		
		i = 0
		while i < len(linesIn):
			polySize = linesIn[i]
			++i
			
			linesOut.append([linesIn[i + j] for j in range(polySize)])
			i += polySize
		
		return Geometry.polyMesh(vertices, linesOut)

	@asyncFunction
	async def planarCut(self, phi = None, normal = None, center = None):
		"""Computes a planar cut of the geometry along either a given plane or a given phi plane"""
		assert phi is not None or normal is not None
		assert phi is None or normal is None
		
		geometry = await self.resolve.asnc()
		
		request = service.GeometryLib.methods.planarCut.Params.newMessage()
		request.geometry = geometry.data
		
		plane = request.plane
		if phi is not None:
			plane.orientation.phi = phi
		if normal is not None:
			plane.orientation.normal = normal
		
		if center is not None:
			plane.center = center
		
		response = await geometryLib().planarCut(request)
		return np.asarray(response.edges).transpose([2, 0, 1])

	@asyncFunction
	async def plotCut(self, phi = 0, ax = None, plot = True, **kwArgs):
		"""Plot the phi cut of a given geometry in either the given axes or the current active matplotlib axes"""
		import matplotlib.pyplot as plt
		import matplotlib
		
		x, y, z = await self.planarCut.asnc(phi = phi)
		r = np.sqrt(x**2 + y**2)
		
		linedata = np.stack([r, z], axis = -1)
		
		coll = matplotlib.collections.LineCollection(linedata, **kwArgs)
		
		if plot:
			if ax is None:
				ax = plt.gca()
			
			ax.add_collection(coll)
		
		return coll

	@asyncFunction
	async def asPyvista(self, reduce: bool = True):
		"""Convert the given geometry into a PyVista / VTK mesh"""
		import numpy as np
		import pyvista as pv
			
		if reduce:
			geometry = await self.reduce.asnc()
		else:
			geometry = await self.merge.asnc()
		
		mergedGeometry = await data.download.asnc(geometry.data.merged)
		
		def extractMesh(entry):
			mesh = entry.mesh
			
			vertexArray = np.asarray(mesh.vertices)
			indexArray	= np.asarray(mesh.indices)
			
			faces = []
			
			if mesh.which_() == 'polyMesh':
				polyRanges = mesh.polyMesh
				
				for iPoly in range(len(polyRanges) - 1):
					start = polyRanges[iPoly]
					end	  = polyRanges[iPoly + 1]
					
					faces.append(end - start)
					faces.extend(indexArray[start:end])
			elif mesh.which_() == 'triMesh':
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
				if tagValue.which_() == 'text':
					mesh.field_data[tagName] = [tagValue.text]
				if tagValue.which_() == 'uInt64':
					mesh.field_data[tagName] = [tagValue.uInt64]
					
			return mesh
		
		dataSets = [
			extractEntry(entry)
			for entry in mergedGeometry.entries
		]
		
		return pv.MultiBlock(dataSets)
	
	@asyncFunction
	async def weightedSample(self, scale: float):
		"""Converts the geometry into a point cloud"""
		resolved = await self.resolve.asnc()
		response = await geometryLib().weightedSample(resolved.data, scale)
		
		return {
			'centers' : np.asarray(response.centers),
			'areas' : response.areas
		}
	
	@staticmethod
	@unstableApi
	def importFrom(filename: str):
		"""Creates a geometry from a file (non-PLY loaded using meshio)"""
		
		if filename.endswith(".ply"):
			return Geometry(native.geometry.readPly(filename), byReference = True)
		
		import meshio
		inputMesh = meshio.read(filename)
		
		all = []
		
		cellsDict = inputMesh.cells_dict
		all = list(cellsDict.get("triangle", [])) + list(cellsDict.get("quad", [])) + list(cellsDict.get("polygon", []))
		
		return Geometry(native.geometry.importRaw(inputMesh.points, all), byReference = True)
	
	@asyncFunction
	@unstableApi
	async def asPolyMesh(self, triangulate: bool = False):
		"""Exports as a single merged polygon mesh"""
		merged = await self.merge.asnc()
		points, polys = await native.geometry.exportRaw(merged.data, triangulate)
		
		return points, polys
		
	
	@asyncFunction
	@unstableApi
	async def exportTo(self, filename: str, binary = True, triangulate = True):
		"""Saves the geometry to the given filename. Supports '.ply', '.stl', and '.vtk' files."""
		
		if filename.endswith(".ply"):
			native.geometry.writePly(await self.getMerged.asnc(), filename, binary)
			return
		
		import meshio
		
		rawPoints, rawPolys = await self.asPolyMesh.asnc(triangulate)
		
		tris = []
		quads = []
		polys = []
		
		for p in rawPolys:
			if len(p) == 2:
				pass
			elif len(p) == 3:
				tris.append(p)
			elif len(p) == 4:
				quads.append(p)
			else:
				polys.append(p)
		
		cells = {}
		if tris:
			cells["triangle"] = tris
		if quads:
			cells["quad"] = quads
		if polys:
			cells["polygon"] = polys
		
		output = meshio.Mesh(points = rawPoints, cells = cells)
		output.write(filename)

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
	mesh.indices = indices
	mesh.polyMesh = [0, 4, 8, 12, 16, 20, 24]
	
	# Publish mesh in distributed data system
	meshRef = data.publish(mesh)
	
	# Put mesh reference & tags into geometry
	return Geometry({
		"mesh" : meshRef,
		"tags" : [
			{"name" : name, "value" : asTagValue(value)}
			for name, value in tags.items()
		]
	})

def geometryLib():
	"""Requests a service.GeometryLib instance from the active backend"""
	return backends.activeBackend().newGeometryLib().pipeline.service
