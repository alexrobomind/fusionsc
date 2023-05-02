from . import service, backends
from .asnc import asyncFunction

from typing import Optional

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
	provider = activeBackend().newHFCamProvider().service
	
	response = await provider.makeToroidalProjection(
		w, h, phi,
		rTarget, zTarget,
		verticalInclination, horizontalInclination, distance,
		viewportHeight, fieldOfView
	)
	return response
	
@asyncFunction
async def make(
	projection : service.HFCamProjection.Reader,
	geometry,
	backend = None,
	edgeTolerance = 0.5,
	depthTolerance = 0.5
):
	provider = activeBackend().newHFCamProvider().service
	
	resolved = await geometry.resolve.asnc()
	cam = provider.makeCamera(projection, resolved.geometry, edgeTolerance, depthTolerance).cam
	return HFCam(cam)

class HFCam:
	def __init__(self, cam):			
		self.cam = cam
	
	@asyncFunction
	def getData(self):
		return self.cam.getData()
	
	@asyncFunction
	async def clear(self):
		await self.cam.clear()
	
	def clone(self):
		return HFCam(self.cam.clone().cam)
	
	@asyncFunction
	async def addPoints(self, points, r, depthTolerance = 0.001):
		req = service.HFCam.methods.addPoints.Params.newMessage()
		req.points = points
		req.r = r
		req.depthTolerance = depthTolerance
		
		await self.cam.addPoints(req)
	
	@asyncFunction
	def get(self):
		return self.cam.get_()
	