from . import native
from .asnc import asyncFunction

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
	
	fieldResolver: List[native.FieldResolver]
	geoResolver: List[native.GeometryResolver]
	
	asyncMode: bool
	
	_grid: native.ToroidalGrid.Reader
	
	def __init__(self, backend, grid):
		self.backend = backend
		self.tracer  = backend.newTracer().service
		
		self.fieldResolvers = []
		self.geoResolvers   = []
		self.asyncMode = False
		self.grid = grid
	
	@property
	def grid(self):
		return self._grid
	
	@grid.setter
	def grid(self, newGrid):
		self._grid      = newGrid.clone()
		self.calculator = self.backend.newFieldCalculator(newGrid).calculator
	
	def importOfflineData(self, filename: str):
		"""
		Loads the data contained in the given offline archives and uses them to
		perform offline resolution.
		"""
		offlineData = native.loadArchive(filename)
		
		self.fieldResolvers.append(native.devices.w7x.offlineCoilsDB(offlineData))
		self.geoResolvers.append(native.devices.w7x.offlineComponentsDB(offlineData))
	
	async def _resolveField(self, field, followRefs: bool = False):
		field = native.readyPromise(field)
		
		for r in self.fieldResolvers:
			try:
				field = await r.resolve(field, followRefs)
			except:
				pass
			
		return field
	
	@optionalAsync
	@asyncFunction
	async def poincareInPhiPlanes(self, points, phiValues, nTurns, config):
		# Resovle & compute field
		resolved    = await self._resolveField(config)
		field       = await self.calculator.compute(resolvedField)
		
		fltResponse = await self.tracer.trace(
			startPoints = points,
			field = resolvedField,
			poincarePlanes = phiValues,
			turnLimit = nTurns
		)
		
		return np.asarray(fltResponse.poincareHits)