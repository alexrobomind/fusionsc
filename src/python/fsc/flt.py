from . import native
from . import data
from .asnc import asyncFunction
from .resolve import resolveField

import numpy as np
import functools

from types import SimpleNamespace

from typing import Optional, List

class FLT:
	# backend: native.RootService.Client
	
	calculator: native.FieldCalculator.Client
	tracer: native.FLT.Client
	
	def __init__(self, backend):
		# self.backend = backend
		self.calculator = backend.newFieldCalculator().calculator
		self.tracer	 = backend.newTracer().service
		self.geometryLib = backend.newGeometryLib().service
	
	def indexGeometry(self, *args, **kwargs):
		"""return self.indexGeometryAsync(*args, **kwargs).wait()"""
		return self.indexGeometryAsync(*args, **kwargs).wait()
	
	def computeField(self, *args, **kwargs):
		"""return self.computeFieldAsync(*args, **kwargs).wait()"""
		return self.computeFieldAsync(*args, **kwargs).wait()
	
	def poincareInPhiPlanes(self, *args, **kwargs):
		"""return self.poincareInPhiPlanesAsync(*args, **kwargs).wait()"""
		return self.poincareInPhiPlanesAsync(*args, **kwargs).wait()
	
	def trace(self, *args, **kwargs):
		return traceAsync(*arg, **kwargs).wait()
	
	def mergeGeometry(self, *args, **kwargs):
		return self.mergeGeometryAsync(*args, **kwargs).wait()
	
	def indexGeometry(self, *args, **kwargs):
		return self.indexGeometryAsync(*args, **kwargs).wait()
	
	def trace(self, *args, **kwargs):
		return self.traceAsync(*args, **kwargs).wait()
	
	@asyncFunction
	async def mergeGeometryAsync(self, geometry):
		resolved  = await geometry.resolve()
		mergedRef = self.geometryLib.merge(resolved.geometry).ref
		
		result = Geometry()
		result.geometry.merged = mergedRef
		return result
	
	@asyncFunction
	async def indexGeometryAsync(self, geometry, grid):
		from . import Geometry
		
		resolved  = await geometry.resolve()
		indexed	  = (await self.geometryLib.index(resolved.geometry, grid)).indexed
		
		result = Geometry()
		result.geometry.indexed = indexed
		return result
	
	@asyncFunction
	async def computeFieldAsync(self, config, grid):
		"""
		Returns an array of shape [3, grid.nPhi, grid.nZ, grid.nR] containing the magnetic field.
		The directions along the first coordinate are phi, z, r
		"""
		import numpy as np
		
		print("Grid:", grid)
		
		resolved = await config.resolve()
		computed = (await self.calculator.compute(resolved.field, grid)).computedField
		fieldData = await data.download(computed.data)
		
		return np.asarray(fieldData).transpose([3, 0, 1, 2])
	
	@asyncFunction
	async def poincareInPhiPlanesAsync(self, points, phiValues, nTurns, config, grid = None, distanceLimit = 1e4, stepSize = 1e-3):
		# Resovle & compute field
		resolvedField = await config.resolve()
		
		if grid is None:
			assert resolvedField.field.which() == 'computed'
			computedField = resolvedField.field.computed
		else:
			computedField = (await self.calculator.compute(resolvedField.field, grid)).computedField
			
		print("Assigning")
		
		# This style is neccessary to convert the start points to an FSC tensor
		request = native.FLTRequest.newMessage()
		request.startPoints = points
		request.field = computedField
		request.poincarePlanes = phiValues
		request.turnLimit = nTurns
		request.distanceLimit = distanceLimit
		request.stepSize = stepSize
		
		print("Tracing")
		fltResponse = await self.tracer.trace(request)
		print("Done")
		
		return np.asarray(fltResponse.poincareHits)
	
	@asyncFunction
	async def traceAsync(self, points, config, geometry = None, grid = None, geometryGrid = None, distanceLimit = 1e4, stepSize = 1e-3, collisionLimit = 0):
		resolvedField = await config.resolve()
		
		if grid is None:
			assert resolvedField.field.which() == 'computed'
			computedField = resolvedField.field.computed
		else:
			computedField = (await self.calculator.compute(resolvedField.field, grid)).computedField
		
		print("Computed field obtained")
		
		if geometry is not None:			
			if geometryGrid is None:
				assert geometry.geometry.which() == 'indexed'
				indexedGeometry = geometry.geometry.indexed
			else:
				indexedGeometry = (await self.indexGeometryAsync(geometry, geometryGrid)).geometry.indexed
				
		print("Indexed geometry obtained")
		
		request = native.FLTRequest.newMessage()
		request.startPoints = points
		request.field = computedField
		request.stepSize = stepSize
		request.distanceLimit = distanceLimit
		request.collisionLimit = collisionLimit
		
		if geometry is not None:
			request.geometry = indexedGeometry
		
		print("Starting trace")
		response = await self.tracer.trace(request)
		
		return SimpleNamespace(
			endPoints = np.asarray(response.endPoints),
			# stopReasons = np.asarray(response.stopReasons)
		)
		