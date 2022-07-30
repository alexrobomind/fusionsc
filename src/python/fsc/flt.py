from . import native
from . import data
from .asnc import asyncFunction
from .resolve import resolveField

import numpy as np
import functools

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
	
	@asyncFunction
	async def indexGeometryAsync(self, geometry, grid):
		resolved  = await geometry.resolve()
		mergedRef = self.geometryLib.merge(resolved).ref
		indexed	  = await self.geometryLib.index(mergedRef, grid)
		
		return indexed
	
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
	async def poincareInPhiPlanesAsync(self, points, phiValues, nTurns, config, grid, distanceLimit = 1e4, stepSize = 1e-3):
		# Resovle & compute field
		resolved	= await config.resolve()
		computed	= (await self.calculator.compute(resolved.field, grid)).computedField
		
		# This style is neccessary to convert the start points to an FSC tensor
		request = native.FLTRequest.newMessage()
		request.startPoints = points
		request.field = computed
		request.poincarePlanes = phiValues
		request.turnLimit = nTurns
		request.distanceLimit = distanceLimit
		request.stepSize = stepSize
		
		fltResponse = await self.tracer.trace(request)
		
		return np.asarray(fltResponse.poincareHits)