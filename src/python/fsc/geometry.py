from . import native

from .asnc import asyncFunction

import numpy as np

def planarCut(*args, **kwargs):
	return planarCutAsync(*args, **kwargs).wait()

@asyncFunction
async def planarCutAsync(mergedGeometryRef, geoLib, phi = None, normal = None, center = None):
	assert phi is not None or normal is not None
	assert phi is None or normal is None
	
	print("Creating request")
	request = native.GeometryLib.methods.planarCut.Params.newMessage()
	request.geoRef = mergedGeometryRef
	
	print("Getting plane")
	plane = request.plane
	
	print("Setting phi")
	if phi is not None:
		plane.orientation.phi = phi
	
	if normal is not None:
		plane.orientation.normal = normal
	
	print("Setting center")
	if center is not None:
		plane.center = center
	
	print("Sending request")
	response = await geoLib.planarCut(request)
	
	return np.asarray(response.edges).transpose([2, 0, 1])