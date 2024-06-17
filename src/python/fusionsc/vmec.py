from . import backends
from . import service
from . import wrappers
from . import native
from . import magnetics
from . import data

from ._api_markers import unstableApi
from .asnc import asyncFunction

import numpy as np

from typing import Optional, List

def _driver():
	return backends.activeBackend().vmecDriver().service

@asyncFunction
@unstableApi
async def sphithetaToPhizr(surfaces : magnetics.SurfaceArray, s, phi, theta, sValues : Optional[List[float]] = None):
	# This also gets caught by the backend but the error message can be confusing
	assert len(surfaces.shape) == 1, "Surface array must be linear"
	
	if sValues is not None:
		assert len(sValues) == surfaces.shape[0], "sValues size must match no. of surfaces"
	
	stacked = np.stack(
		np.broadcast(s, phi, theta),
		axis = 0
	)
	
	response = await _driver().computePositions(surfaces.data, stacked, sValues = sValues)
	phi, z, r = np.asarray(response.PhiZR)
	return phi, z, r

@asyncFunction
@unstableApi
async def phizrToSphitheta(surfaces : magnetics.SurfaceArray, phi, z, r, sValues : Optional[List[float]] = None):
	# This also gets caught by the backend but the error message can be confusing
	assert len(surfaces.shape) == 1, "Surface array must be linear"
	
	if sValues is not None:
		assert len(sValues) == surfaces.shape[0], "sValues size must match no. of surfaces"
	
	stacked = np.stack(
		np.broadcast(phi, z, r),
		axis = 0
	)
	
	response = await _driver().invertPositions(surfaces.data, stacked, sValues = sValues)
	
	s, phi, theta = np.asarray(response.sPhiTheta)
	return s, phi, theta

@asyncFunction
@unstableApi
async def writeMGrid(field: magnetics.MagneticConfig, path: str, grid: Optional[service.ToroidalGrid.ReaderOrBuilder] = None):
	computed = await field.compute.asnc(grid)
	await native.vmec.writeMGrid(computed.data.computed, path)

@unstableApi
class VmecEquilibrium(wrappers.structWrapper(service.VmecResult)):		
	@property
	def surfaces(self):
		return magnetics.SurfaceArray(self.data.surfaces, byReference = True)
	
	@property
	def volume(self):
		return self.data.volume
	
	@property
	def energy(self):
		return self.data.energy		
	
	@surfaces.setter
	def surfaces(self, arr):
		self.data.surfaces = arr.data
	
	@staticmethod
	def fromWoutNc(filename: str):
		return VmecEquilibrium(native.vmec.loadOutput(filename))

@unstableApi
class Request(wrappers.structWrapper(service.VmecRequest)):
	@asyncFunction
	async def submit(self):
		return Response(await _driver().run(self.data))

@unstableApi
class Response(wrappers.structWrapper(service.VmecResponse)):
	@property
	def ok(self):
		return self.data.which_() == "ok"
	
	@asyncFunction
	async def getResult(self):
		if self.data.which_() == "failed":
			raise ValueError("Run failed " + self.data.failed)
		
		return VmecEquilibrium(await data.download(self.data.ok))
