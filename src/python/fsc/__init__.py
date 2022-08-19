from . import native
from . import devices
from . import flt
from . import asnc
from . import data
from . import geometry

from .native import kj
from .native import capnp

from .asnc import run, asyncFunction, wait, Promise
from .resolve import importOfflineData

from .native.timer import delay

from typing import Optional

__all__ = [
	'run', 'asyncFunction', 'wait', 'importOfflineData', 'delay', 'Promise', 'MagneticConfig'
]

# Initialize event loop for main thread
asnc.startEventLoop()

localServer = native.LocalRootServer()
localRoot = localServer.connect()

def local() -> native.RootService:
	"""
	Creates a local instance of the FSC services
	"""
	
	return localRoot # localServer.connect() #native.connectSameThread()

def tracer(backend: Optional[native.RootService] = None) -> flt.FLT:
	"""
	Creates a new field line tracer backed by an FSC service. If
	no backend is specified, creates a local backend.
	"""
	
	if backend is None:
		backend = local()
			
	return flt.FLT(backend)

class Geometry:
	def __init__(self, geo = None):
		self._holder = native.Geometry.newMessage()
		
		if geo is None:
			self._holder.initNested()
		else:
			self._holder.nested = geo
	
	@property
	def geometry(self):
		return self._holder.nested
	
	@geometry.setter
	def geometry(self, newVal):
		self._holder.nested = newVal
	
	def ptree(self):
		import printree
		printree.ptree(self.geometry)
	
	def graph(self, **kwargs):
		return capnp.visualize(self.geometry, **kwargs)
		
	def __repr__(self):
		return str(self.geometry)
		
	@asyncFunction
	async def resolve(self):
		return Geometry(await resolve.resolveGeometry.asnc(self.geometry))
	
	def __add__(self, other):
		if not isinstance(other, Geometry):
			return NotImplemented
			
		result = Geometry()
		
		if self.geometry.which() == 'combined' and other.geometry.which() == 'combined':
			result.geometry.combined = list(self.geometry.combined) + list(other.geometry.combined)
			return result
		
		if self.geometry.which() == 'combined':
			result.geometry.combined = list(self.geometry.combined) + [other.geometry]
			return result
		
		if other.geometry.which() == 'combined':
			result.geometry.combined = [self.geometry] + list(other.geometry.combined)
			return result
		
		result.geometry.combined = [self.geometry, other.geometry]
		return result
	
	def translate(self, dx):
		result = Geometry()
		
		transformed = result.geometry.initTransformed()
		shifted = transformed.initShifted()
		shifted.shift = dx
		shifted.node.leaf = self.geometry
		
		return result
	
	def rotate(self, angle, axis, center = [0, 0, 0]):
		result = Geometry()
		
		transformed = result.geometry.initTransformed()
		turned = transformed.initTurned()
		turned.axis = axis
		turned.angle = angle
		turned.center = center
		turned.node.leaf = self.geometry
		
		return result
		
	
	
class MagneticConfig:
	"""
	Magnetic configuration class. Wraps an instance of fsc.native.MagneticField.Builder
	and provides access to +, -, *, and / operators.
	"""
	
	_holder: native.MagneticField.Builder
	
	def __init__(self, field = None):	
		self._holder = native.MagneticField.newMessage()
		
		if field is None:
			self._holder.initNested()
		else:
			self._holder.nested = field
	
	# Directly assigning python variables does not copy them, so we
	# need to do a bit of propery magic to make sure we assign
	# struct fields	
	@property
	def field(self):
		return self._holder.nested
	
	@field.setter
	def field(self, newVal):
		self._holder.nested = newVal
	
	@asyncFunction
	async def resolve(self):
		return MagneticConfig(await resolve.resolveField.asnc(self.field))
	
	def ptree(self):
		import printree
		printree.ptree(self.field)
	
	def graph(self, **kwargs):
		return capnp.visualize(self.field, **kwargs)
		
	def __repr__(self):
		return str(self.field)
	
	def __neg__(self):
		result = MagneticConfig()
		
		# Remove double inversion
		if self.field.which() == 'invert':
			result.field = self.field.invert
			return result
		
		result.field.invert = self.field
		return result
	
	def __add__(self, other):
		if not isinstance(other, MagneticConfig):
			return NotImplemented()
			
		result = MagneticConfig()
		
		if self.field.which() == 'sum' and other.field.which() == 'sum':
			result.field.sum = list(self.field.sum) + list(other.field.sum)
			return result
		
		if self.field.which() == 'sum':
			result.field.sum = list(self.field.sum) + [other.field]
			return result
		
		if other.field.which() == 'sum':
			result.field.sum = [self.field] + list(other.field.sum)
		
		result.field.sum = [self.field, other.field]
		
		return result
		
	def __sub__(self, other):
		return self + (-other)
	
	def __mul__(self, factor):
		import numbers
		
		if not isinstance(factor, numbers.Number):
			return NotImplemented
		
		result = MagneticConfig()
		
		if self.field.which() == 'scaleBy':
			result.field.scaleBy = self.field.scaleBy
			result.field.scaleBy.factor *= factor
		
		scaleBy = result.field.initScaleBy()
		scaleBy.field = self.field
		scaleBy.factor = factor
		
		return result
	
	def __rmul__(self, factor):
		return self.__mul__(factor)
	
	def __truediv__(self, divisor):
		return self * (1 / divisor)
	