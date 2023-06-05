"""Frontend module for magnetic field calculations"""

from . import data
from . import service
from . import capnp
from . import resolve
from . import backends
from . import efit
from . import wrappers

from .asnc import asyncFunction

from typing import Optional

class CoilFilament(wrappers.structWrapper(service.Filament)):
	"""
	Set of coils that can be associated with a current to compute magnetic fields.
	"""
	
	@asyncFunction
	async def resolve(self):
		"""Resolves all coils contained in the coil description"""
		return CoilFilament(await resolve.resolveFilament.asnc(self.data))
	
	def __add__(self, other):
		if isinstance(other, int) and other == 0:
			return self
		
		if not isinstance(other, CoilFilament):
			return NotImplemented()
			
		result = CoilFilament()
		
		if self.data.which() == 'sum' and other.data.which() == 'sum':
			result.data.sum = list(self.data.sum) + list(other.data.sum)
			return result
		
		if self.data.which() == 'sum':
			result.data.sum = list(self.data.sum) + [other.data]
			return result
		
		if other.data.which() == 'sum':
			result.data.sum = [self.data] + list(other.data.sum)
		
		result.data.sum = [self.data, other.data]
		
		return result
	
	def __radd__(self, other):
		return self.__add__(other)
	
	def biotSavart(self, width: float = 0.01, stepSize: float = 0.01, current: float = 1, windingNo: int = 1):
		"""Creates a magnetic field by applying the BiotSavart law to the contained coil filaments"""
		result = MagneticConfig()
		
		bs = result.field.initFilamentField()
		bs.current = current
		bs.biotSavartSettings.stepSize = stepSize
		bs.biotSavartSettings.width = width
		bs.filament = self.data
		bs.windingNo = windingNo
		
		return result
	
	@staticmethod
	def fromArray(data):
		"""Creates a coil from numpy array of shape [3, nPoints]"""
		data = np.asarray(data)
		
		# Validate shape
		assert len(data.shape) == 2
		assert data.shape[0] == 3
		
		# Transpose input (internal coils have shape [nPoints, 3])
		data = data.T
		
		# Publish data as ref
		filament = service.Filament.newMessage()
		filament.inline = data
		
		ref = data.publish(filament)
		
		result = CoilFilament()
		result.data.ref = ref
		
		return result

class MagneticConfig(wrappers.structWrapper(service.MagneticField)):
	"""
	Magnetic configuration class. Wraps an instance of fusionsc.service.MagneticField.Builder
	and provides access to +, -, *, and / operators.
	"""
	
	@asyncFunction
	async def resolve(self):
		"""Resolves contained coils and magnetic configurations contained in this object (returned in a new instance)"""
		return MagneticConfig(await resolve.resolveField.asnc(self.data))
	
	@asyncFunction
	async def compute(self, grid):
		"""Computes the magnetic field on the specified grid."""
		if grid is None:
			assert self.data.which() == 'computedField', 'Must specify grid or use pre-computed field'
			return MagneticConfig(self.data)
		
		result = MagneticConfig()
		
		resolved = await self.resolve.asnc()
		
		backend = backends.activeBackend()
		calculator = backend.newFieldCalculator().service
		
		comp = result.data.initComputedField()
		comp.grid = grid
		comp.data = calculator.compute(resolved.data, grid).computedField.data
		
		return result
	
	def __neg__(self):
		result = MagneticConfig()
		
		# Remove double inversion
		if self.data.which() == 'invert':
			result.data = self.data.invert
			return result
		
		result.data.invert = self.data
		return result
	
	def __add__(self, other):
		if isinstance(other, int) and other == 0:
			return self
		
		if not isinstance(other, MagneticConfig):
			return NotImplemented()
			
		result = MagneticConfig()
		
		if self.data.which() == 'sum' and other.data.which() == 'sum':
			result.data.sum = list(self.data.sum) + list(other.data.sum)
			return result
		
		if self.data.which() == 'sum':
			result.data.sum = list(self.data.sum) + [other.data]
			return result
		
		if other.data.which() == 'sum':
			result.data.sum = [self.data] + list(other.data.sum)
		
		result.data.sum = [self.data, other.data]
		
		return result
	
	def __radd__(self, other):
		return self.__add__(other)
		
	def __sub__(self, other):
		return self + (-other)
	
	def __mul__(self, factor):
		import numbers
		
		if not isinstance(factor, numbers.Number):
			return NotImplemented
		
		result = MagneticConfig()
		
		if self.data.which() == 'scaleBy':
			result.data.scaleBy = self.data.scaleBy
			result.data.scaleBy.factor *= factor
		
		scaleBy = result.data.initScaleBy()
		scaleBy.field = self.data
		scaleBy.factor = factor
		
		return result
	
	def __rmul__(self, factor):
		return self.__mul__(factor)
	
	def __truediv__(self, divisor):
		return self * (1 / divisor)
	
	@staticmethod
	def fromEFit(contents: Optional[str] = None, filename: Optional[str] = None):
		"""Creates a magnetic equilibrium field form an EFit file"""
		assert contents or filename, "Must provide either GEqdsk file contents or filename"
		
		if contents is None:
			with open(filename, "r") as f:
				contents = f.read()
			
		return MagneticConfig({'axisymmetricEquilibrium' : efit.eqFromGFile(contents)})


@asyncFunction
async def visualizeCoils(field):
	"""Convert the given geometry into a PyVista / VTK mesh"""
	import numpy as np
	import pyvista as pv
	
	coils = []
	
	async def processCoil(coil):
		if coil.which() == 'inline':
			coils.append(np.asarray(coil.inline))
			return
		
		if coil.which() == 'ref':
			local = await data.download.asnc(coil.ref)
			await processCoil(local)
			return
		
		if coil.which() == 'nested':
			await processCoil(coil.nested)
			return
		
		print("Warning: Unresolved node can not be visualized")
		print(coil)
			
	
	async def process(field):
		if field.which() == 'sum':
			for x in field.sum:
				await process(x)
			return
		
		if field.which() == 'ref':
			local = await data.download(field.ref)
			await process(local)
			return
		
		if field.which() == 'scaleBy':
			await process(field.scaleBy.field)
			return
		
		if field.which() == 'invert':
			await process(field.invert)
			return
		
		if field.which() == 'nested':
			await process(field.nested)
			return
		
		if field.which() == 'cached':
			await process(field.cached.nested)
			return
		
		if field.which() == 'filamentField':
			await processCoil(field.filamentField.filament)
			return
		
		print("Warning: Unresolved nodes can not be visualized")
	
	resolved = await field.resolve.asnc()
	await process(resolved.data)
		
	def makeCoil(coil):
		result = pv.lines_from_points(coil)
		return result
		
	dataSets = [
		makeCoil(coil)
		for coil in coils
	]
		
	return pv.MultiBlock(dataSets)