"""Frontend module for magnetic field calculations"""

from . import data
from . import service
from . import capnp
from . import resolve
from . import backends
from . import efit
from . import wrappers

from .asnc import asyncFunction

import numpy as np
import copy

from typing import Optional

def _calculator():
	return backends.activeBackend().newFieldCalculator().pipeline.service

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
		
		if self.data.which_() == 'sum' and other.data.which_() == 'sum':
			result.data.sum = list(self.data.sum) + list(other.data.sum)
			return result
		
		if self.data.which_() == 'sum':
			result.data.sum = list(self.data.sum) + [other.data]
			return result
		
		if other.data.which_() == 'sum':
			result.data.sum = [self.data] + list(other.data.sum)
		
		result.data.sum = [self.data, other.data]
		
		return result
	
	def __radd__(self, other):
		return self.__add__(other)
	
	def biotSavart(self, width: float = 0.01, stepSize: float = 0.01, current: float = 1, windingNo: int = 1):
		"""Creates a magnetic field by applying the BiotSavart law to the contained coil filaments"""
		result = MagneticConfig()
		
		bs = result.data.initFilamentField()
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
		"""
		Computes the magnetic field on the specified grid. Doesn't download the field to the local machine.
		
		Returns:
			A magnetic field of type 'computedField' that holds a reference to the (eventually computed,
			possible remotely held) computed field.
		
		"""
		if grid is None:
			assert self.data.which_() == 'computedField', 'Must specify grid or use pre-computed field'
			return MagneticConfig(self.data)
		
		result = MagneticConfig()
		
		resolved = await self.resolve.asnc()
		
		comp = result.data.initComputedField()
		comp.grid = grid
		comp.data = _calculator().compute(resolved.data, grid).pipeline.computedField.data
		
		return result
	
	def __await__(self):
		assert self.data.which_() == 'computedField', 'Can only await computed fields'
		return self.data.computedField.data.__await__()
	
	@asyncFunction
	async def interpolateXyz(self, points, grid = None):
		"""
		Evaluates the magnetic field in the given coordinates. Outside the grid, the field will use the constant
		values (in slab coordinates) at the grid boundary.
	
		Parameters:
			- points: A numpy-array of shape [3, ...] (at least 1D) with the points in x, y, z coordinates.
			- grid: An optional grid parameter required if the field is not yet computed. The grid
		
		Returns:
			A numpy array of shape points.shape with the field as x, y, z field (cartesian).
		"""
		compField = (await self.compute.asnc(grid)).data.computedField
		
		response = await _calculator().evaluateXyz(compField, points)
		return np.asarray(response.values)
	
	@asyncFunction
	async def getComputed(self):
		"""
		For a field of type "computed", returns the grid and the downloaded field tensor on the grid.
		
		Returns:
			- A service.ToroidalGrid.Builder describing the grid.
			- A tensor of shape [nPhi, nZ, nR, 3] holding the magnetic field. The last axis describes
			  the magnetic field component. The components are (indices 0 to 2) bPhi, bZ, bR.
		"""
		assert self.data.which_() == 'computedField', 'Can only download fields for which the compute operation was initialized (returned by .compute)'
		computed = self.data.computedField
		
		return copy.copy(computed.grid), np.asarray(await data.download.asnc(computed.data))
	
	def __neg__(self):
		result = MagneticConfig()
		
		# Remove double inversion
		if self.data.which_() == 'invert':
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
		
		if self.data.which_() == 'sum' and other.data.which_() == 'sum':
			result.data.sum = list(self.data.sum) + list(other.data.sum)
			return result
		
		if self.data.which_() == 'sum':
			result.data.sum = list(self.data.sum) + [other.data]
			return result
		
		if other.data.which_() == 'sum':
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
		
		if self.data.which_() == 'scaleBy':
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
	
	@staticmethod
	def fromComputed(grid, field):
		"""
		Creates a field by loading a computed field.
		
		Parameters:
			- grid: An object that can be assigned to a service.MagneticField (e.g. YAML string, service.MagneticField.Reader or .Builder, apprioriate python dict)
			- field: A tensor of shape [nPhi, nZ, nR, 3] holding the magnetic field. The last axis describes
			  the magnetic field component. The components are (indices 0 to 2) bPhi, bZ, bR.
		
		Returns:
			A magnetic field object matching the given grid and the field values at the interpolation points
		"""
		tensorData = service.Float64Tensor.newMessage(field)
		return MagneticConfig({'computedField' : {
			'grid' : grid,
			'data' : data.publish(tensorData)
		}})


@asyncFunction
async def visualizeCoils(field):
	"""Convert the given geometry into a PyVista / VTK mesh"""
	import numpy as np
	import pyvista as pv
	
	coils = []
	
	async def processCoil(coil):
		if coil.which_() == 'inline':
			coils.append(np.asarray(coil.inline))
			return
		
		if coil.which_() == 'ref':
			local = await data.download.asnc(coil.ref)
			await processCoil(local)
			return
		
		if coil.which_() == 'nested':
			await processCoil(coil.nested)
			return
		
		print("Warning: Unresolved node can not be visualized")
		print(coil)
			
	
	async def process(field):
		if field.which_() == 'sum':
			for x in field.sum:
				await process(x)
			return
		
		if field.which_() == 'ref':
			local = await data.download(field.ref)
			await process(local)
			return
		
		if field.which_() == 'scaleBy':
			await process(field.scaleBy.field)
			return
		
		if field.which_() == 'invert':
			await process(field.invert)
			return
		
		if field.which_() == 'nested':
			await process(field.nested)
			return
		
		if field.which_() == 'cached':
			await process(field.cached.nested)
			return
		
		if field.which_() == 'filamentField':
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