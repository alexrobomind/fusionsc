from . import native

from .asnc import asyncFunction

import numpy as np

def planarCut(*args, **kwargs):
	return planarCutAsync(*args, **kwargs).wait()

@asyncFunction
async def planarCutAsync(geometry, geoLib, phi = None, normal = None, center = None):
	assert phi is not None or normal is not None
	assert phi is None or normal is None
	
	assert geometry.geometry.which() in ['indexed', 'merged']
	
	print("Creating request")
	request = native.GeometryLib.methods.planarCut.Params.newMessage()
	if geometry.geometry.which() == 'merged':
		request.geoRef = geometry.geometry.merged
	
	if geometry.geometry.which() == 'indexed':
		request.geoRef = geometry.geometry.indexed.base
	
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