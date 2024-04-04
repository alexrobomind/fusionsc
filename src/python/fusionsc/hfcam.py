"""Synthetic camera diagnostic to load distributions from impact point clouds"""

from . import service, backends, wrappers
from .asnc import asyncFunction

from typing import Optional

class Projection(wrappers.structWrapper(service.HFCamProjection)):
	pass

def _provider():
	return backends.activeBackend().newHFCamProvider().pipeline.service
	

@asyncFunction
async def toroidalProjection(
	w: int,
	h: int,
	phi: float,
	rTarget: float,
	zTarget : float,
	verticalInclination: float,
	horizontalInclination: float,
	distance: float,
	viewportHeight: float,
	fieldOfView: float
):
	"""Computes a new camera projection setup that looks at the given target point"""
	
	response = await _provider().makeToroidalProjection(
		w, h, phi,
		rTarget, zTarget,
		verticalInclination, horizontalInclination, distance,
		viewportHeight, fieldOfView
	)
	return Projection(response)
	
@asyncFunction
async def make(
	projection : Projection,
	geometry,
	edgeTolerance = 0.5,
	depthTolerance = 0.5
):
	"""Creates a new HF camera based on the given projection"""	
	resolved = await geometry.resolve.asnc()
	cam = _provider().makeCamera(projection.data, resolved.data, edgeTolerance, depthTolerance).pipeline.cam
	return HFCam(cam)

class HFCam:
	def __init__(self, cam):			
		self.cam = cam
	
	@asyncFunction
	def getData(self):
		"""Downloads the data (projection and buffers) backing this camera instance"""
		return self.cam.getData()
	
	@asyncFunction
	async def clear(self):
		"""Resets the heat-flux accumulator buffer to 0"""
		await self.cam.clear()
	
	def clone(self):
		"""Creates a new camera which is an exact copy of this one (including the heat flux buffer contents)"""
		return HFCam(self.cam.clone().pipeline.cam)
	
	@asyncFunction
	async def addPoints(self, points, r, depthTolerance = 0.001):
		"""
		Converts the given point cloud of energy packets into a 2D heat flux distribution and adds it to the accumulator buffer.
		
		Parameters:
			- points: Array-like of shape [3, ...] containing the haet samples
			- r: Radius of each heat packet (in m)
			- depthTolerance: How far points are allowed to sit behind the geometry surface to still be considered 'visible'
		"""
		req = service.HFCam.methods.addPoints.Params.newMessage()
		req.points = points
		req.r = r
		req.depthTolerance = depthTolerance
		
		await self.cam.addPoints(req)
	
	@asyncFunction
	def get(self):
		"""
		Obtains the contents of the heat flux accumulator buffer.
		"""
		return self.cam.get()
	