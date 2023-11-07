from . import backends
from . import service
from . import wrappers

import numpy as np

def _driver():
	return backends.activeBackend().vmecDriver().service

def newRequest():
	return service.VmecRequest.newMessage()

@asyncFunction
async def run(request):	
	return await _driver().run(request)

@asyncFunction
async def sphithetaToPhizr(surfaces, s, phi, theta):
	stacked = np.stack(
		np.broadcast(s, phi, theta),
		axis = 0
	)
	
	response = await _driver().computePositions(surfaces, stacked)
	phi, z, r = np.asarray(response.PhiZR)
	return phi, z, r

@asyncFunction
async def phizrToSphitheta(surfaces, phi, z, r):
	stacked = np.stack(
		np.broadcast(phi, z, r),
		axis = 0
	)
	
	response = await _driver().invertPositions(surfaces, stacked)
	
	s, phi, theta = np.asarray(response.sPhiTheta)
	return s, phi, theta