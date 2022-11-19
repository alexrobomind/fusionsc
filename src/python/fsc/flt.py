"""
Root service frontend
"""

from . import native
from . import data
from .asnc import asyncFunction
from .resolve import resolveField

import numpy as np
import functools

from types import SimpleNamespace

from typing import Optional, List

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

class FLT:
	# backend: native.RootService.Client
	
	calculator: native.FieldCalculator.Client
	tracer: native.FLT.Client
	
	def __init__(self, backend):
		# self.backend = backend
		self.calculator = backend.newFieldCalculator().service
		self.tracer	 = backend.newTracer().service
		self.geometryLib = backend.newGeometryLib().service
	
	@asyncFunction
	async def mergeGeometry(self, geometry):
		resolved  = await geometry.resolve.asnc()
		mergedRef = self.geometryLib.merge(resolved.geometry).ref
		
		result = Geometry()
		result.geometry.merged = mergedRef
		return result
	
	@asyncFunction
	async def indexGeometry(self, geometry, grid):
		from . import Geometry
		
		resolved  = await geometry.resolve.asnc()
		indexed	  = (await self.geometryLib.index(resolved.geometry, grid)).indexed
		
		result = Geometry()
		result.geometry.indexed = indexed
		return result
	
	@asyncFunction
	async def computeField(self, config, grid):
		"""
		Returns an array of shape [3, grid.nPhi, grid.nZ, grid.nR] containing the magnetic field.
		The directions along the first coordinate are phi, z, r
		"""
		import numpy as np
		
		print("Grid:", grid)
		
		resolved = await config.resolve.asnc()
		computed = (await self.calculator.compute(resolved.field, grid)).computedField
		fieldData = await data.download.asnc(computed.data)
		
		return np.asarray(fieldData).transpose([3, 0, 1, 2])
	
	@asyncFunction
	async def poincareInPhiPlanes(self, points, phiPlanes, turnLimit, config, **kwArgs):
		result = await self.trace.asnc(points, config, turnLimit = turnLimit, phiPlanes = phiPlanes, **kwArgs)
		return result["poincareHits"]
	
	@asyncFunction
	async def connectionLength(self, points, config, geometry, **kwargs):
		result = await self.trace.asnc(points, config, geometry = geometry, collisionLimit = 1, **kwargs)
		return result["endPoints"][3]
		
	@asyncFunction
	async def trace(self,
		points, config,
		geometry = None,
		grid = None, geometryGrid = None, 
		
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
			assert resolvedField.field.which() == 'computed', 'Can only omit grid if field is pre-computed'
			computedField = resolvedField.field.computed
		else:
			computedField = (await self.calculator.compute(resolvedField.field, grid)).computedField
		
		if geometry is not None:	
			if geometryGrid is None:
				assert geometry.geometry.which() == 'indexed', 'Can only omit geometry grid if geometry is already indexed'
				indexedGeometry = geometry.geometry.indexed
			else:
				indexedGeometry = (await self.indexGeometry.asnc(geometry, geometryGrid)).geometry.indexed
		
		request = native.FLTRequest.newMessage()
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
			"responseSize" : native.capnp.totalSize(response)
		}