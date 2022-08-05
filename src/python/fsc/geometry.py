from . import native

from .asnc import asyncFunction

import numpy as np

def localGeoLib():
	return native.connectSameThread().newGeometryLib().service

def planarCut(*args, **kwargs):
	return planarCutAsync(*args, **kwargs).wait()

@asyncFunction
async def planarCutAsync(geometry, phi = None, normal = None, center = None):
	assert phi is not None or normal is not None
	assert phi is None or normal is None
	
	geometry = await geometry.resolve()
	
	request = native.GeometryLib.methods.planarCut.Params.newMessage()
	request.geometry = geometry.geometry
	
	plane = request.plane
	if phi is not None:
		plane.orientation.phi = phi
	if normal is not None:
		plane.orientation.normal = normal
	
	if center is not None:
		plane.center = center
	
	response = await localGeoLib().planarCut(request)
	return np.asarray(response.edges).transpose([2, 0, 1])