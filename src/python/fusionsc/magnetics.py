from . import data
from . import service
from . import capnp
from . import resolve
from . import backends
from . import efit

from .asnc import asyncFunction

from typing import Optional

class CoilFilament:
	"""
	Set of coils that can be associated with a current to compute magnetic fields.
	"""
	
	_holder: service.Filament.Builder
	
	def __init__(self, filament = None):
		self._holder = service.Filament.newMessage()
		
		if filament is None:
			self._holder.initNested()
		else:
			self._holder.nested = filament
	
	@property
	def filament(self):
		return self._holder.nested
	
	@filament.setter
	def filament(self, newVal):
		self._holder.nested = newVal
	
	@asyncFunction
	async def resolve(self):
		return CoilFilament(await resolve.resolveFilament(self.filament))
	
	def ptree(self):
		import printree
		printree.ptree(self.field)
	
	def graph(self, **kwargs):
		return capnp.visualize(self.field, **kwargs)
		
	def __repr__(self):
		return str(self.filament)
	
	def __add__(self, other):
		if isinstance(other, int) and other == 0:
			return self
		
		if not isinstance(other, CoilFilament):
			return NotImplemented()
			
		result = CoilFilament()
		
		if self.filament.which() == 'sum' and other.filament.which() == 'sum':
			result.filament.sum = list(self.filament.sum) + list(other.filament.sum)
			return result
		
		if self.filament.which() == 'sum':
			result.filament.sum = list(self.filament.sum) + [other.filament]
			return result
		
		if other.filament.which() == 'sum':
			result.filament.sum = [self.filament] + list(other.filament.sum)
		
		result.filament.sum = [self.filament, other.filament]
		
		return result
	
	def __radd__(self, other):
		return self.__add__(other)
	
	def biotSavart(self, width: float = 0.01, stepSize: float = 0.01, current: float = 1, windingNo: int = 1):
		result = MagneticConfig()
		
		bs = result.field.initFilamentField()
		bs.current = current
		bs.biotSavartSettings.stepSize = stepSize
		bs.biotSavartSettings.width = width
		bs.filament = self.filament
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
		result.filament.ref = ref
		
		return result

class MagneticConfig:
	"""
	Magnetic configuration class. Wraps an instance of fusionsc.service.MagneticField.Builder
	and provides access to +, -, *, and / operators.
	"""
	
	_holder: service.MagneticField.Builder
	
	def __init__(self, field = None):	
		self._holder = service.MagneticField.newMessage()
		
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
	
	@asyncFunction
	async def compute(self, grid):
		if grid is None:
			assert self.field.which() == 'computedField', 'Must specify grid or use pre-computed field'
			return MagneticConfig(self.field)
		
		result = MagneticConfig()
		
		resolved = await self.resolve.asnc()
		
		backend = backends.activeBackend()
		calculator = backend.newFieldCalculator().service
		
		comp = result.field.initComputedField()
		comp.grid = grid
		comp.data = calculator.compute(resolved.field, grid).computedField.data
		
		return result
	
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
		if isinstance(other, int) and other == 0:
			return self
		
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
	
	def __radd__(self, other):
		return self.__add__(other)
		
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
	
	@asyncFunction
	@staticmethod
	async def load(filename):
		field = await data.readArchive.asnc(filename)
		return MagneticConfig(field)
	
	@asyncFunction
	async def save(self, filename):
		await data.writeArchive.asnc(self.field, filename)
	
	@staticmethod
	def fromEFit(contents: Optional[str] = None, filename: Optional[str] = None):
		assert contents or filename, "Must provide either GEqdsk file contents or filename"
		
		if contents is None:
			with open(filename, "r") as f:
				contents = f.read()
			
		result = MagneticConfig()
		result.field.axisymmetricEquilibrium = axisymmetricEquilibrium = efit.eqFromGFile(contents)
		
		return result


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
			local = await data.download(coil.ref)
			await processCoil(local)
			return
		
		if coil.which() == 'nested':
			await processCoil(coil.nested)
			return
		
		print("Warning: Unresolved nodes can not be visualized")
			
	
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
	await process(resolved.field)
	
	def makeCoil(coil):
		vertexArray = np.asarray(coil)
		nPoints = vertexArray.shape[0]
		
		indices = [nPoints] + list(range(nPoints))
		
		return pv.PolyData(vertexArray, lines = np.asarray(indices))
		
	dataSets = [
		makeCoil(coil)
		for coil in coils
	]
	
	return pv.MultiBlock(dataSets)