"""
Root service frontend
"""
from . import data
from . import service
from . import capnp
from . import resolve
from . import inProcess
from . import efit

from .asnc import asyncFunction
from .resolve import resolveField

import numpy as np
import functools

from types import SimpleNamespace

from typing import Optional, List

class MagneticConfig:
	"""
	Magnetic configuration class. Wraps an instance of fusionsc.service.MagneticField.Builder
	and provides access to +, -, *, and / operators.
	"""
	
	_holder: service.MagneticField.Builder
	
	def __init__(self, field = None):	
		self._holder = service.MagneticField.newMessage()
		
		if field is None:
			self._holder.initNested()
		else:
			self._holder.nested = field
	
	# Directly assigning python variables does not copy them, so we
	# need to do a bit of propery magic to make sure we assign
	# struct fields	
	@property
	def field(self):
		return self._holder.nested
	
	@field.setter
	def field(self, newVal):
		self._holder.nested = newVal
	
	@asyncFunction
	async def resolve(self):
		return MagneticConfig(await resolve.resolveField.asnc(self.field))
	
	def ptree(self):
		import printree
		printree.ptree(self.field)
	
	def graph(self, **kwargs):
		return capnp.visualize(self.field, **kwargs)
		
	def __repr__(self):
		return str(self.field)
	
	def __neg__(self):
		result = MagneticConfig()
		
		# Remove double inversion
		if self.field.which() == 'invert':
			result.field = self.field.invert
			return result
		
		result.field.invert = self.field
		return result
	
	def __add__(self, other):
		if not isinstance(other, MagneticConfig):
			return NotImplemented()
			
		result = MagneticConfig()
		
		if self.field.which() == 'sum' and other.field.which() == 'sum':
			result.field.sum = list(self.field.sum) + list(other.field.sum)
			return result
		
		if self.field.which() == 'sum':
			result.field.sum = list(self.field.sum) + [other.field]
			return result
		
		if other.field.which() == 'sum':
			result.field.sum = [self.field] + list(other.field.sum)
		
		result.field.sum = [self.field, other.field]
		
		return result
		
	def __sub__(self, other):
		return self + (-other)
	
	def __mul__(self, factor):
		import numbers
		
		if not isinstance(factor, numbers.Number):
			return NotImplemented
		
		result = MagneticConfig()
		
		if self.field.which() == 'scaleBy':
			result.field.scaleBy = self.field.scaleBy
			result.field.scaleBy.factor *= factor
		
		scaleBy = result.field.initScaleBy()
		scaleBy.field = self.field
		scaleBy.factor = factor
		
		return result
	
	def __rmul__(self, factor):
		return self.__mul__(factor)
	
	def __truediv__(self, divisor):
		return self * (1 / divisor)
	
	@asyncFunction
	@staticmethod
	async def load(filename):
		field = await data.readArchive.asnc(filename)
		return MagneticConfig(field)
	
	@asyncFunction
	async def save(self, filename):
		await data.writeArchive.asnc(self.field, filename)
	
	@staticmethod
	def fromEFit(contents: Optional[str] = None, filename: Optional[str] = None):
		assert contents or filename, "Must provide either GEqdsk file contents or filename"
		
		if contents is None:
			with open(filename, "r") as f:
				contents = f.read()
			
		result = MagneticConfig()
		result.field.axisymmetricEquilibrium = axisymmetricEquilibrium = efit.eqFromGFile(contents)
		
		return result

def symmetrize(points, nSym = 1, stellaratorSymmetric = False):
	x, y, z = points
	phi = np.arctan2(y, x)
	phi = phi % (2 * np.pi / nSym)
	
	r = np.sqrt(x**2 + y**2)
	
	phi = np.stack([phi + dPhi for dPhi in np.linspace(0, 2 * np.pi, nSym, endpoint = False)], axis = 0)
	z = np.stack([z] * nSym, axis = 0)
	r = np.stack([r] * nSym, axis = 0)
	
	if stellaratorSymmetric:
		phi = np.concatenate([phi, -phi], axis = 0)
		z	= np.concatenate([z, -z], axis = 0)
		r	= np.concatenate([r, r], axis = 0)
	
	x = r * np.cos(phi)
	y = r * np.sin(phi)
	
	return np.stack([x, y, z], axis = 0)

@asyncFunction
async def visualizeMapping(mapping, nPoints = 50):
	import numpy as np
	import pyvista as pv
	
	mapping = await data.download.asnc(mapping)
	
	def vizFilament(filament):
		filData = np.asarray(filament.data).reshape([-1, 6])
				
		if len(filData) == 0:
			return pv.MultiBlock([])
			
		
		phi = np.linspace(filament.phiStart, filament.phiEnd, filament.nIntervals + 1)
		r = filData[:, 0]
		z = filData[:, 1]
		
		x = np.cos(phi) * r
		y = np.sin(phi) * r
		
		xyz = np.stack([x, y, z], axis = 1)
		
		objects = []
		
		spline = pv.Spline(xyz, nPoints)
		objects.append(spline)
		
		for i in []: #range(0, filament.nIntervals):
			phi0 = phi[i]
			r0 = filData[i, 0]
			z0 = filData[i, 1]
			
			for j in [0, 1]:
				dr = filData[i, 2 + 2 * j]
				dz = filData[i, 2 + 2 * j + 1]
				
				x0 = r0 * np.cos(phi0)
				y0 = r0 * np.sin(phi0)
				
				dx = dr * np.cos(phi0)
				dy = dr * np.sin(phi0)
				
				arrow = pv.Arrow(start = [x0, y0, z0], direction = [dx, dy, dz], scale = 'auto')
				objects.append(arrow)
		
		return pv.MultiBlock(objects)
	
	return pv.MultiBlock([
		vizFilament(f)
		for filArray in [mapping.fwd.filaments, mapping.bwd.filaments]
		for f in filArray
	])

@asyncFunction
async def visualizeCoils(field):
	"""Convert the given geometry into a PyVista / VTK mesh"""
	import numpy as np
	import pyvista as pv
	
	coils = []
	
	async def processCoil(coil):
		if coil.which() == 'inline':
			coils.append(np.asarray(coil.inline))
			return
		
		if coil.which() == 'ref':
			local = await data.download(coil.ref)
			await processCoil(local)
			return
		
		if coil.which() == 'nested':
			await processCoil(coil.nested)
			return
		
		print("Warning: Unresolved nodes can not be visualized")
			
	
	async def process(field):
		if field.which() == 'sum':
			for x in field.sum:
				await process(x)
			return
		
		if field.which() == 'ref':
			local = await data.download(field.ref)
			await process(local)
			return
		
		if field.which() == 'scaleBy':
			await process(field.scaleBy.field)
			return
		
		if field.which() == 'invert':
			await process(field.invert)
			return
		
		if field.which() == 'nested':
			await process(field.nested)
			return
		
		if field.which() == 'cached':
			await process(field.cached.nested)
			return
		
		if field.which() == 'filamentField':
			await processCoil(field.filamentField.filament)
			return
		
		print("Warning: Unresolved nodes can not be visualized")
	
	resolved = await field.resolve.asnc()
	await process(resolved.field)
	
	def makeCoil(coil):
		vertexArray = np.asarray(coil)
		nPoints = vertexArray.shape[0]
		
		indices = [nPoints] + list(range(nPoints))
		
		return pv.PolyData(vertexArray, lines = np.asarray(indices))
		
	dataSets = [
		makeCoil(coil)
		for coil in coils
	]
	
	return pv.MultiBlock(dataSets)

class FLT:
	backend: service.RootService.Client
	
	def __init__(self, backend = None):
		if backend is None:
			backend = inProcess.root()
			
		self.backend = backend
	
	@property
	def calculator(self):
		return self.backend.newFieldCalculator().service
	
	@property
	def tracer(self):
		return self.backend.newTracer().service
	
	@property
	def geometryLib(self):
		return self.backend.newGeometryLib().service
	
	@property
	def mapper(self):
		return self.backend.newMapper().service
	
	@asyncFunction
	async def fieldValues(self, config, grid):
		"""
		Returns an array of shape [3, grid.nPhi, grid.nZ, grid.nR] containing the magnetic field.
		The directions along the first coordinate are phi, z, r
		"""
		import numpy as np
		
		field = await self.computeField.asnc(config, grid)
		fieldData = await data.download.asnc(field.field.computedField.data)
		
		return np.asarray(fieldData).transpose([3, 0, 1, 2])
	
	@asyncFunction
	async def computeField(self, config, grid):		
		resolved = await config.resolve.asnc()
		computed = (await self.calculator.compute(resolved.field, grid)).computedField
		
		result = MagneticConfig()
		result.field.computedField = computed
		
		return result
	
	@asyncFunction
	async def poincareInPhiPlanes(self, points, phiPlanes, turnLimit, config, **kwArgs):
		result = await self.trace.asnc(points, config, turnLimit = turnLimit, phiPlanes = phiPlanes, **kwArgs)
		return result["poincareHits"]
	
	@asyncFunction
	async def connectionLength(self, points, config, geometry, **kwargs):
		result = await self.trace.asnc(points, config, geometry = geometry, collisionLimit = 1, **kwargs)
		return result["endPoints"][3]
	
	
	@asyncFunction
	async def computeMapping(self,
		startPoints, config, grid = None, 
		nSym = 1, stepSize = 0.001, batchSize = 1000,
		nPhi = 30, filamentLength = 5, cutoff = 1,
		dx = 0.001
	):
		resolvedField = await config.resolve.asnc()
		
		if grid is None:
			assert resolvedField.field.which() == 'computedField', 'Can only omit grid if field is pre-computed'
			computedField = resolvedField.field.computedField
		else:
			computedField = (await self.calculator.compute(resolvedField.field, grid)).computedField
		
		# We use the request based API because tensor values are not yet supported for fields
		request = service.MappingRequest.newMessage()
		request.startPoints = startPoints
		request.field = computedField
		request.dx = dx
		request.filamentLength = filamentLength
		request.cutoff = cutoff
		request.nPhi = nPhi
		request.nSym = nSym
		request.stepSize = stepSize
		request.batchSize = batchSize
		
		response = self.mapper.computeMapping(request)
		
		return response.mapping
		
	@asyncFunction
	async def trace(self,
		points, config,
		geometry = None,
		grid = None, geometryGrid = None,
		mapping = None,
		
		# Limits to stop tracing
		distanceLimit = 1e4, turnLimit = 0, stepLimit = 0, stepSize = 1e-3, collisionLimit = 0,
		
		# Plane intersections
		phiPlanes = [],
		
		# Diffusive transport specification
		isotropicDiffusionCoefficient = None,
		parallelConvectionVelocity = None, parallelDiffusionCoefficient = None,
		meanFreePath = 1, meanFreePathGrowth = 0
	):
		"""Performs a custom trace with user-supplied parameters"""
		assert parallelConvectionVelocity is None or parallelDiffusionCoefficient is None
		if isotropicDiffusionCoefficient is not None: 
			assert parallelConvectionVelocity is not None or parallelDiffusionCoefficient is not None
		
		resolvedField = await config.resolve.asnc()
		
		if grid is None:
			assert resolvedField.field.which() == 'computedField', 'Can only omit grid if field is pre-computed'
			computedField = resolvedField.field.computedField
		else:
			computedField = (await self.calculator.compute(resolvedField.field, grid)).computedField
		
		if geometry is not None:	
			if geometryGrid is None:
				assert geometry.geometry.which() == 'indexed', 'Can only omit geometry grid if geometry is already indexed'
				indexedGeometry = geometry.geometry.indexed
			else:
				indexedGeometry = (await self._indexGeometry.asnc(geometry, geometryGrid)).geometry.indexed
		
		request = service.FLTRequest.newMessage()
		request.startPoints = points
		request.field = computedField
		request.stepSize = stepSize
		
		request.distanceLimit = distanceLimit
		request.stepLimit = stepLimit
		request.collisionLimit = collisionLimit
		request.turnLimit = turnLimit
				
		# Diffusive transport model
		if isotropicDiffusionCoefficient is not None:
			request.perpendicularModel.isotropicDiffusionCoefficient = isotropicDiffusionCoefficient
			
			request.parallelModel.meanFreePath = meanFreePath
			request.parallelModel.meanFreePathGrowth = meanFreePathGrowth
			
			if parallelConvectionVelocity is not None:
				request.parallelModel.convectiveVelocity = parallelConvectionVelocity
			else:
				request.parallelModel.diffusionCoefficient = parallelDiffusionCoefficient
		
		if len(phiPlanes) > 0:
			planes = request.initPlanes(len(phiPlanes))
			
			for plane, phi in zip(planes, phiPlanes):
				plane.orientation.phi = phi
		
		if geometry is not None:
			request.geometry = indexedGeometry
		
		if mapping is not None:
			request.mapping = mapping
		
		response = await self.tracer.trace(request)
		
		endTags = {
			str(tagName) : tagData
			for tagName, tagData in zip(response.tagNames, np.asarray(response.endTags))
		}
		
		return {
			"endPoints" : np.asarray(response.endPoints),
			"poincareHits" : np.asarray(response.poincareHits),
			"stopReasons" : np.asarray(response.stopReasons),
			"endTags" : endTags,
			"responseSize" : capnp.totalSize(response)
		}
	
	@asyncFunction
	async def findAxis(self, field, grid = None, startPoint = None, stepSize = 0.001, nTurns = 10, nIterations = 10):
		resolvedField = await field.resolve.asnc()
		
		if grid is None:
			assert resolvedField.field.which() == 'computedField', 'Can only omit grid if field is pre-computed'
			computedField = resolvedField.field.computedField
			grid = computedField.grid
		else:
			computedField = (await self.calculator.compute(resolvedField.field, grid)).computedField
		
		# If start point is not provided, use grid center
		if startPoint is None:
			startPoint = [0.5 * (grid.rMax + grid.rMin),
				0, 0.5 * (grid.zMax + grid.zMin)
			]
		
		request = service.FindAxisRequest.newMessage()
		request.startPoint = startPoint
		request.field = computedField
		request.stepSize = stepSize
		request.nTurns = nTurns
		request.nIterations = nIterations
		
		response = await self.tracer.findAxis(request)
		
		return np.asarray(response.pos), np.asarray(response.axis)
	
	@asyncFunction
	async def axisCurrent(self, field, current, grid = None, startPoint = None, stepSize = 0.001, nTurns = 10, nIterations = 10):
		_, axis = await self.findAxis(field, grid, startPoint, stepSize, nTurns, nIterations)
		
		result = MagneticField()
		
		filField = result.initFilamentField()
		filField.current = current
		filField.biotSavartSettings.stepSize = stepSize
		filField.filament.inline = axis
		
		return result
	
	@asyncFunction
	async def _mergeGeometry(self, geometry):
		resolved  = await geometry.resolve.asnc()
		mergedRef = self.geometryLib.merge(resolved.geometry).ref
		
		result = Geometry()
		result.geometry.merged = mergedRef
		return result
	
	@asyncFunction
	async def _indexGeometry(self, geometry, grid):
		from . import Geometry
		
		resolved  = await geometry.resolve.asnc()
		indexed	  = (await self.geometryLib.index(resolved.geometry, grid)).indexed
		
		result = Geometry()
		result.geometry.indexed = indexed
		return result
	