from . import backends
from . import service
from . import wrappers
from . import native
from . import magnetics
from . import data

from .wrappers import unstableApi
from .asnc import asyncFunction

import numpy as np

def _driver():
	return backends.activeBackend().vmecDriver().service

@asyncFunction
@unstableApi
async def sphithetaToPhizr(surfaces, s, phi, theta):
	stacked = np.stack(
		np.broadcast(s, phi, theta),
		axis = 0
	)
	
	response = await _driver().computePositions(surfaces, stacked)
	phi, z, r = np.asarray(response.PhiZR)
	return phi, z, r

@asyncFunction
@unstableApi
async def phizrToSphitheta(surfaces, phi, z, r):
	stacked = np.stack(
		np.broadcast(phi, z, r),
		axis = 0
	)
	
	response = await _driver().invertPositions(surfaces, stacked)
	
	s, phi, theta = np.asarray(response.sPhiTheta)
	return s, phi, theta

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
