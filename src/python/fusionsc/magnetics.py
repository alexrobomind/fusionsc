"""Frontend module for magnetic field calculations"""

from . import data
from . import service
from . import capnp
from . import resolve
from . import backends
from . import efit
from . import wrappers

from .asnc import asyncFunction
from ._api_markers import unstableApi

import numpy as np
import copy

from typing import Optional, Sequence, Literal

def _calculator():
	return backends.activeBackend().newFieldCalculator().pipeline.service

class SurfaceArray(wrappers.structWrapper(service.FourierSurfaces)):
	"""A wrapper around service.FourierSurfaces.Builder that exposes array-like behavior"""
	
	@property
	def shape(self):
		return list(self.data.rCos.shape)[:-2]
	
	def apply(self, op, *extraArgs):
		args = [self] + list(extraArgs)
		
		d = self.data
		result = service.FourierSurfaces.newMessage(d)
		
		result.rCos = op(*[np.asarray(arg.data.rCos) for arg in args])
		result.zSin = op(*[np.asarray(arg.data.zSin) for arg in args])
		
		if d.which_() == 'nonSymmetric':
			result.nonSymmetric.rSin = op(*[np.asarray(arg.data.nonSymmetric.rSin) for arg in args])
			result.nonSymmetric.zCos = op(*[np.asarray(arg.data.nonSymmetric.zCos) for arg in args])
		
		return SurfaceArray(result)
	
	def __getitem__(self, sl):
		if isinstance(sl, tuple):
			slEx = sl + (slice(None), slice(None))
		else:
			slEx = (sl, slice(None), slice(None))
		
		def op(x):
			return x.__getitem__(slEx)
		
		return self.apply(op)
	
	def __add__(self, other):
		if not isinstance(other, SurfaceArray):
			return NotImplemented
		
		def op(a, b):
			return a + b
		
		return self.apply(op, other)
	
	def __sub__(self, other):
		if not isinstance(other, SurfaceArray):
			return NotImplemented
			
		def op(a, b):
			return a - b
		
		return self.apply(op, other)
	
	def __mul__(self, l):
		# Broadcast to right shape
		bc = np.asarray(l)[..., None, None]
		
		def op(x):
			return bc * x
		
		return self.apply(op)
	
	def __rmul__(self, l):
		return self.__mul__(l)
	
	def __truediv__(self, l):
		# Broadcast to right shape
		bc = np.asarray(l)[..., None, None]
		
		def op(x):
			return x / bc
		
		return self.apply(op)
	
	def __rtruediv__(self, l):
		return self.__truediv__(l)
	
	@asyncFunction
	async def evaluate(self, phi: Sequence[float], theta: Sequence[float]):
		response = await _calculator().evalFourierSurface(self.data, phi, theta)
		
		return {
			'points' : np.asarray(response.points),
			'phiDerivatives' : np.asarray(response.phiDerivatives),
			'thetaDerivatives' : np.asarray(response.thetaDerivatives)
		}
	
	def asGeometry(self, nPhi = 100, nTheta = 100, radialShift = 0):
		from . import geometry
		pipeline = _calculator().surfaceToMesh(self.data, nPhi, nTheta, radialShift).pipeline
		
		return geometry.Geometry({"merged" : pipeline.merged})

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
	def fromArray(coilData):
		"""Creates a coil from numpy array of shape [3, nPoints]"""
		coilData = np.asarray(coilData)
		
		# Validate shape
		assert len(coilData.shape) == 2
		assert coilData.shape[0] == 3
		
		# Transpose input (internal coils have shape [nPoints, 3])
		coilData = coilData.T
		
		# Publish data as ref
		filament = service.Filament.newMessage()
		filament.inline = coilData
		
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
	
	@asyncFunction
	async def computeCached(self, grid):
		"""
		Attaches a cached computed version of this field. In future computations, this field will be interpolated
		as long as all points of the compute request lie inside its grid definition.
		"""
		
		computed = await self.compute.asnc(grid)
		return MagneticConfig({
			"cached" : {
				"nested" : self.data,
				"computed" : computed.data.computedField
			}
		})
	
	def __await__(self):
		assert self.data.which_() == 'computedField', 'Can only await computed fields'
		return self.data.computedField.data.__await__()
	
	def translate(self, dx):
		"""Returns a new field shifted by the given vector"""
		result = MagneticConfig()
		
		transformed = result.data.initTransformed()
		shifted = transformed.initShifted()
		shifted.shift = dx
		
		if self.data.which_() == "transformed":
			shifted.node = self.data.transformed
		else:
			shifted.node.leaf = self.data
		
		return result
	
	def rotate(self, angle, axis, center = [0, 0, 0]):
		"""Returns a new field rotated around the prescribed axis and center point"""
		result = MagneticConfig()
		
		transformed = result.data.initTransformed()
		turned = transformed.initTurned()
		turned.axis = axis
		turned.angle.rad = angle
		turned.center = center
		
		if self.data.which_() == "transformed":
			turned.node = self.data.transformed
		else:
			turned.node.leaf = self.data
		
		return result
	
	@asyncFunction
	async def interpolateXyz(self, points, grid = None):
		"""
		Evaluates the magnetic field at target positions by first computing the magnetic field
		at the target points (if not yet done), and then 
	
		Parameters:
			- points: A numpy-array of shape [3, ...] (at least 1D) with the points in x, y, z coordinates.
			- grid: An optional grid parameter required if the field is not yet computed. The grid
		
		Returns:
			A numpy array of shape points.shape with the field as x, y, z field (cartesian).
		"""
		compField = (await self.compute.asnc(grid)).data.computedField
		
		response = await _calculator().interpolateXyz(compField, points)
		return np.asarray(response.values)
	
	@asyncFunction
	async def evaluateXyz(self, points):
		"""
		Evaluates the magnetic field in the given coordinates. Unlike interpolateXyz, this function
		does NOT compute the field on a grid and then interpolate, but instead evaluates the field
		directly at the given point.
	
		Parameters:
			- points: A numpy-array of shape [3, ...] (at least 1D) with the points in x, y, z coordinates.
		
		Returns:
			A numpy array of shape points.shape with the field as x, y, z field (cartesian).
		"""
		resolved = await self.resolve.asnc()
		response = await _calculator().evaluateXyz(resolved.data, points)
		return np.asarray(response.values)
	
	@asyncFunction
	async def evaluatePhizr(self, points):
		"""
		Evaluates the magnetic field in the given coordinates. 
		
		Parameters:
			- points: A numpy-array of shape [3, ...] (at least 1D) with the points in r, z, phi coordinates.
		
		Returns:
			A numpy array of shape points.shape with the field as x, y, z field (cartesian).
		"""
		resolved = await self.resolve.asnc()
		response = await _calculator().evaluatePhizr(resolved.data, points)
		return np.asarray(response.values)
	
	@asyncFunction
	async def getComputed(self, grid = None):
		"""
		For a field of type "computed", returns the grid and the downloaded field tensor on the grid.
		
		Returns:
			- A service.ToroidalGrid.Builder describing the grid.
			- A tensor of shape [nPhi, nZ, nR, 3] holding the magnetic field. The last axis describes
			  the magnetic field component. The components are (indices 0 to 2) bPhi, bZ, bR.
		"""
		computed = await self.compute.asnc(grid)
		cf = computed.data.computedField
		
		return copy.copy(cf.grid), np.asarray(await data.download.asnc(cf.data))
		
	@asyncFunction
	@unstableApi
	async def calculateRadialModes(
		self, surfaces: SurfaceArray,
		normalizeAgainst : "Optional[MagneticConfig]" = None,
		nMax = 5, mMax = 5, nPhi = 0, nTheta = 0, nSym = 1,
		useFFT = True,
		quantity: Literal["field", "flux"] = "field"
	):
		"""
		Calculates the radial Fourier modes of this field (or the ratio to the given
		background field) on the given surfaces
		
		Parameters:
			- surfaces: Description of the magnetic surfaces to evaluate on.
			- normalizeAgainst: Optional field to normalize again
			- nMax: Maximum absolute toroidal mode number to calculate
			- mMax: Maximum poloidal mode number to calcualte
			- nPhi: Number of toroidal points on grid
			- nTheta: Number of poloidal points on grid
			- nSym: Toroidal symmetry
			- useFFT: Whether to use the fast FFT-based Fourier path (as opposed to a slower cosine fit)
			- quantity: Whether to calculate the radial field or the radial flux (field * (dx/dPhi x dx/dTheta))
		
		Returns:
			A dict with the following entries:
			- cosCoeffs: [..., 2 * nMax + 1, mMax + 1] array of cosine coefficients of Fourier mode expansion
			- sinCoeffs: [..., 2 * nMax + 1, mMax + 1] array of sine coefficients of Fourier mode expansion
			- nTor: [2 * nMax + 1, 1] array of toroidal mode numbers
			- mPol: [1, mMax + 1] array of poloidal mode numbers
			
		"""
		resolved = await self.resolve.asnc()
		
		background = None
		if normalizeAgainst is not None:
			background = (await normalizeAgainst.resolve.asnc()).data
		
		if nPhi == 0:
			nPhi = 2 * nMax + 1
		
		if nTheta == 0:
			nTheta = 2 * mMax + 1
		
		if (nPhi != 2 * nMax + 1 or nTheta != 2 * mMax + 1) and useFFT:
			import warnings
			warnings.warn("calculateRadialModes can only use the FFT fast path if nPhi == 2 * nMax + 1 and nTheta == 2 * mMax + 1. Other values are not recommended")
		
		response = await _calculator().calculateRadialModes(
			resolved.data, background,
			surfaces.data,
			nMax, mMax,
			nPhi, nTheta,
			nSym,
			useFFT,
			quantity
		)
		
		return {
			"cosCoeffs" : np.asarray(response.cosCoeffs),
			"sinCoeffs" : np.asarray(response.sinCoeffs),
			"reCoeffs" : np.asarray(response.reCoeffs),
			"imCoeffs" : np.asarray(response.imCoeffs),
			"radialValues" : np.asarray(response.radialValues),
			"nTor" : np.asarray(response.nTor),
			"mPol" : np.asarray(response.mPol),
			"phi" : np.asarray(response.phi),
			"theta" : np.asarray(response.theta)
		}
	
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
	
	@staticmethod
	def fromDipoles(positions, moments, radii):
		positions = np.asarray(positions)
		moments = np.asarray(moments)
		radii = np.asarray(radii)
		
		assert len(positions.shape) == 2
		assert len(moments.shape) == 2
		assert positions.shape[0] == 3
		assert moments.shape[0] == 3
		assert positions.shape[1] == moments.shape[1]
		assert len(radii) == positions.shape[1]
		
		return MagneticConfig({'dipoleCloud' : {
			'positions' : positions,
			'magneticMoments' : moments,
			'radii' : radii
		}})

@asyncFunction
async def extractCoils(field):
	import numpy as np
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
		
		if coil.which_() == 'sum':
			for e in coil.sum:
				await processCoil(e)
			return
		
		print("Warning: Unresolved node can not be extracted")
		print(coil)
			
	
	async def processField(field):
		if field.which_() == 'sum':
			for x in field.sum:
				await processField(x)
			return
		
		if field.which_() == 'ref':
			local = await data.download(field.ref)
			await processField(local)
			return
		
		if field.which_() == 'scaleBy':
			if field.scaleBy.factor != 0:
				await processField(field.scaleBy.field)
				
			return
		
		if field.which_() == 'invert':
			await processField(field.invert)
			return
		
		if field.which_() == 'nested':
			await processField(field.nested)
			return
		
		if field.which_() == 'cached':
			await processField(field.cached.nested)
			return
		
		if field.which_() == 'filamentField':
			if field.filamentField.current * field.filamentField.windingNo != 0:
				await processCoil(field.filamentField.filament)
			return
		
		print("Warning: Unresolved nodes can not be visualized")
	
	async def processAny(x):
		if hasattr(x, "_fusionsc_wraps"):
			await processAny(x._fusionsc_wraps)
			
		if isinstance(x, list):
			for e in x:
				await processAny(e)
			
		if isinstance(x, dict):
			for e in x.values():
				await processAny(e)
		
		if isinstance(x, service.MagneticField.ReaderOrBuilder):
			await processField(await resolve.resolveField.asnc(x))
		
		if isinstance(x, service.Filament.ReaderOrBuilder):
			await processCoil(await resolve.resolveFilament.asnc(x))
	
	await processAny(field)
	
	return coils

@asyncFunction
async def visualizeCoils(field):
	"""Convert the given geometry into a PyVista / VTK mesh"""
	import pyvista as pv
	
	coils = await extractCoils.asnc(field)
		
	def makeCoil(coil):
		result = pv.lines_from_points(coil)
		return result
		
	dataSets = [
		makeCoil(coil)
		for coil in coils
	]
		
	return pv.MultiBlock(dataSets)

"""
def _fourierSurfaces(rCos, zSin, rsin = None, zCos = None, nSym = 1, nTurns = 1) -> service.FourierSurfaces.Builder:
	result = service.FourierSurfaces.newMessage()
	
	assert rCos is not None
	assert zSin is not None
	
	assert zSin.shape == rCos.shape
	
	nToroidalModes = rCos.shape[-2]
	nPoloidalModes = rCos.shape[-1]
	
	assert nPoloidalModes >= 1
	assert nToroidalModes >= 1
	assert nToroidalModes % 2 == 1
	
	result.toroidalSymmetry = nSym
	result.nTurns = nTurns
	
	result.mPol = nPoloidalModes - 1
	result.nTor = nToroidalModes // 2
		
	result.rCos = rCos
	result.zSin = zSin
	
	if rSin is not None or zCos is not None:
		assert rSin is not None and zCos is not None
		
		assert rSin.shape == rCos.shape
		assert zCos.shape == rCos.shape
		
		nonSym = result.initNonSymmetric()
		nonSym.rSin = rSin
		nonSym.zCos = zCos
	
	return result
"""