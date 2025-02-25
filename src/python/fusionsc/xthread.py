"""
Allows cross-thread sharing of FusionSC objects.

Usage:
  # This object is bound to the original thread
  obj = ...
  
  # This object can be freely passed between threads
  handle = fusionsc.xthread.export(obj)
  
  ...
  
  # This object is bound to the thread in which get() was called
  objInNewThread = handle.get()
"""

import typing
import asyncio

from . import backends, serialize, data

from .wrappers import asyncFunction

@asyncFunction
async def export(val: typing.Any, download: bool = False):
	"""
	Creates a representation of the target object that can be shared across
	python threads.
	"""
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
