"""Synthetic camera diagnostic to load distributions from impact point clouds"""

from . import service, backends, wrappers
from .asnc import asyncFunction
import numpy as np

def _kd():
	return backends.activeBackend().newKDTreeService().pipeline.service

def _localKd():
	return backends.localBackend().newKDTreeService().pipeline.service

class KDTree(wrappers.RefWrapper):
	@staticmethod
	def build(points):
		svc = _kd()
		
		r = service.KDTreeService.methods.buildSimple.Params.newMessage()
		r.points = np.asarray(points)
		
		return KDTree(svc.buildSimple(r).pipeline.ref)
	
	@asyncFunction
	async def sample(self, scale = 0):
		svc = _kd()
		
		response = await _kd().sample(self.ref, scale)
		return np.asarray(response.points)
	
	@asyncFunction
	async def plot(self, dims = [0, 1], pointThreshold = 1e-3, maxLevel = -1):
		import matplotlib.pyplot as plt
		
		data = await self.download.asnc()
		
		chunkSize = data.chunkSize
		
		points = []
		
		def getNode(i):
			iChunk = i // chunkSize
			offset = i % chunkSize
			
			chunk = data.chunks[iChunk]
			
			return chunk.nodes[offset], np.asarray(chunk.boundingBoxes)[offset, dims]
		
		def plotNode(i, level):
			node, bounds = getNode(i)
		
			if node.which_() == "interior" and level != maxLevel:
				for i in range(node.interior.start, node.interior.end):
					plotNode(i, level + 1)
			
			diam = np.linalg.norm(bounds[:, 1] - bounds[:, 0])
			
			x0 = bounds[0, 0]
			x1 = bounds[0, 1]
			y0 = bounds[1, 0]
			y1 = bounds[1, 1]
			
			if diam >= pointThreshold:
				plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], c = f"C{level}")
			else:
				plt.scatter(x0, y0, c = f"C{level}")
		
		plotNode(0, 0)
					
			
		
