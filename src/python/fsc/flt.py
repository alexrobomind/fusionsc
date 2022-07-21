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
	_holder: native.MagneticField.Builder
	
	def __init__(self):	
		self._holder = native.MagneticField.newMessage()
		self._holder.initNested()
	
	# Directly assigning python variables does not copy them, so we
	# need to do a bit of propery magic to make sure we assign
	# struct fields	
	@property
	def field(self):
		return self._holder.nested
	
	@field.setter
	def field(self, newVal):
		self._holder.nested = newVal
	
	def __neg__(self):
		result = Config()
		
		# Remove double inversion
		if self.field.which() == 'invert':
			result.field = self.field.invert
			return result
		
		result.field.invert = self.field
		return result
	
	def __add__(self, other):
		result = Config()
		
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
	
	def __repr__(self):
		return str(self.field)

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
		resolved    = await resolveField(config.field)
		computed    = (await self.calculator.compute(resolvedField, grid)).computedField
		
		fltResponse = await self.tracer.trace(
			startPoints = points,
			field = resolvedField,
			poincarePlanes = phiValues,
			turnLimit = nTurns
		)
		
		return np.asarray(fltResponse.poincareHits)