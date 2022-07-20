from . import native
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

class Config:
	field: native.MagneticField.Builder
	
	def __init__():
		field = native.MagneticField.newMessage()
	
	def __neg__(self):
		result = Config()
		result.field.setInvert(self.field)
		return result
	
	def __add__(self, other):
		result = Config()
		sum = result.initSum(2)
		sum[0] = self.field
		sum[1] = other.field
		return result
	
	def __sub__(self, other):
		return self + (-other)
		

class FLT:
	backend: native.RootService.Client
	
	calculator: native.FieldCalculator.Client
	tracer: native.FLT.Client
		
	asyncMode: bool
	
	def __init__(self, backend):
		self.backend = backend
		self.calculator = self.backend.newFieldCalculator().calculator
		self.tracer  = backend.newTracer().service
		
		self.asyncMode = False
	
	@optionalAsync
	@asyncFunction
	async def poincareInPhiPlanes(self, points, phiValues, nTurns, config, grid):
		# Resovle & compute field
		resolved    = await resolveField(config)
		computed    = (await self.calculator.compute(resolvedField, grid)).computedField
		
		fltResponse = await self.tracer.trace(
			startPoints = points,
			field = resolvedField,
			poincarePlanes = phiValues,
			turnLimit = nTurns
		)
		
		return np.asarray(fltResponse.poincareHits)