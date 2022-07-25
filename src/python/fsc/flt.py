from . import native
from . import data
from .asnc import asyncFunction
from .resolve import resolveField

import numpy as np
import functools

from typing import Optional, List

def optionalAsync(f):
	"""
	Wrapper for functions that have switchable behavior based on
	asyncMode. Will wait on returned promise if asyncMode is False
	"""
	@functools.wraps(f)
	def wrapper(self, *args, **kwargs):
		if not self.asyncMode:
			return f(self, *args, **kwargs).wait()
		
		return f(self, *args, **kwargs)
	
	return wrapper


class FLT:
	# backend: native.RootService.Client
	
	calculator: native.FieldCalculator.Client
	tracer: native.FLT.Client
		
	asyncMode: bool
	
	def __init__(self, backend):
		# self.backend = backend
		self.calculator = backend.newFieldCalculator().calculator
		self.tracer  = backend.newTracer().service
		
		self.asyncMode = False
	
	@optionalAsync
	@asyncFunction
	async def computeField(self, config, grid):
		import numpy as np
		
		resolved = await config.resolve()
		computed = (await self.calculator.compute(resolved.field, grid)).computedField
		fieldData = await data.download(computed.data)
		
		return np.asarray(fieldData)
	
	@optionalAsync
	@asyncFunction
	async def poincareInPhiPlanes(self, points, phiValues, nTurns, config, grid):
		# Resovle & compute field
		resolved    = await config.resolve()
		computed    = (await self.calculator.compute(resolved.field, grid)).computedField
		
		fltResponse = await self.tracer.trace(
			startPoints = points,
			field = resolvedField,
			poincarePlanes = phiValues,
			turnLimit = nTurns
		)
		
		return np.asarray(fltResponse.poincareHits)