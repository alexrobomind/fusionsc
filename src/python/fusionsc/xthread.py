"""Gives access to native resources of the fusionsc library"""
import typing
import asyncio

from . import backends, serialize, data

from .wrappers import asyncFunction

@asyncFunction
async def export(val: typing.Any, download: bool = False):
	ref = data.publish(val)
	putResponse = await backends.localResources().put(ref, download)
	
	return XThreadHandle(putResponse.id)

class XThreadHandle:
	_id: int
	
	def __init__(self, id):
		self._id = id
		
	def getRef(self):
		return backends.localResources().get(self._id).pipeline.ref
	
	@asyncFunction
	async def get(self):
		val = await data.download.asnc(self.getRef(), unwrapMode = "strict")
		return val
	
	def __del__(self):
		asyncio.run(backends.localResources().erase(self._id))
