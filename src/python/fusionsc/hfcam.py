from . import service, inProcess
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
	provider = inProcess.root().newHFCamProvider().service
	
	response = await provider.makeToroidalProjection(
		w, h, phi,
		rTarget, zTarget,
		verticalInclination, horizontalInclination, distance,
		viewportHeight, fieldOfView
	)
	return response

class HFCam:
	def __init__(self, data, backend):			
		self.backend = backend
		self.data = data
	
	@property
	def provider(self):
		return self.backend.newHFCamProvider().service
	
	@asyncFunction
	@staticmethod
	async def prepare(
		projection : service.HFCamProjection.Reader,
		geometry,
		backend = None,
		edgeTolerance = 0.5,
		depthTolerance = 0.5
	):
		if backend is None:
			backend = inProcess.root()
		
		provider = backend.newHFCamProvider().service
		
		resolved = await geometry.resolve.asnc()
		data = await provider.makeCamera(projection, resolved.geometry, edgeTolerance, depthTolerance)
		return HFCam(data, backend)