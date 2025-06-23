"""
Functions for field-line tracing and interpretation of traces.
"""
from . import data
from . import service
from . import capnp
from . import resolve
from . import backends
from . import efit
from . import magnetics
from . import wrappers
from . import geometry
from . import serialize

from .asnc import asyncFunction
from ._api_markers import unstableApi

import numpy as np
import functools
import numbers

from types import SimpleNamespace

from typing import Optional, List

@serialize.cls()
class FieldlineMapping(wrappers.RefWrapper):
	"""
	Interface class for field line mappings.
	"""
	
	@asyncFunction
	async def mapGeometry(self,
		geometry: geometry.Geometry,
		toroidalSymmetry: int = 1,
		nPhi: int = 1, nU: int = 10, nV: int = 10
	):
		"""
		Pre-computes a field-aligned geometry mapping that can be used to enable
		large step sizes in mapping-based tracing.
		
		Note: Straight line geometry might acquire curved shape during this configuration.
		If your geometry is coarse, consider refining the mesh with Geometry.triangulate(...).
		
		Parameters:
			- geometry: Input geometry to be transformed .
			- nSym: Toroidal symmetry of the geometry.
			- nPhi, nU, nV: Number of cells (per axis) in the collision grid index. U and V
			  are the mapping coordinates.
		"""
		resolved = await geometry.resolve.asnc()
		response = await _mapper().mapGeometry(self.ref, resolved.data, toroidalSymmetry, nPhi, nU, nV)
		return MappingWithGeometry(response.mapping)

class MappingWithGeometry(wrappers.structWrapper(service.GeometryMapping)):
	@asyncFunction
	async def getSection(self, index: int):
		"""
		Extract and download the mapping geometry for a geometry section.
		"""
		response = await _mapper().getSectionGeometry(self.data, index)
		return geometry.Geometry({"indexed" : response.geometry})

def symmetrize(points, nSym = 1, stellaratorSymmetric = False):
	"""
	Takes a point-cloud and creates a new point cloud obeying the prescribed discrete
	symmetry by applying all corresponding discrete transformations (phi-shifts and flips).
	
	Parameters:
		- points: A numpy-array of shape [3, ...] (at least 1D) with the points in x, y, z coordinates.
		- nSym: Toroidal symmetry to be applied to the cloud.
		- stellaratorSymmetric: Whether to apply the Stellarator symmetry (phi -> -phi, z -> -z, r -> r) as well.
	
	Returns:
		An array of shape [3, nTotalSym] + points.shape[1:] containing the new point cloud
		with nTotalSym = 2 * nSym if stellaratorSymmetric else nSym
	"""
	x, y, z = points
	phi = np.arctan2(y, x)
	phi = phi % (2 * np.pi / nSym)
	
	r = np.sqrt(x**2 + y**2)
	
	phi = np.stack([phi + dPhi for dPhi in np.linspace(0, 2 * np.pi, nSym, endpoint = False)], axis = 0)
	z = np.stack([z] * nSym, axis = 0)
	r = np.stack([r] * nSym, axis = 0)
	
	if stellaratorSymmetric:
		phi = np.concatenate([phi, -phi], axis = 0)
		z	= np.concatenate([z, -z], axis = 0)
		r	= np.concatenate([r, r], axis = 0)
	
	x = r * np.cos(phi)
	y = r * np.sin(phi)
	
	return np.stack([x, y, z], axis = 0)

def _tracer():
	return backends.activeBackend().newTracer().pipeline.service

def _mapper():
	return backends.activeBackend().newMapper().pipeline.service

@asyncFunction
async def fieldValues(config, grid):
	"""
	DEPRECATED & SLATED FOR REMOVAL IN 3.0
	
	Use fusionsc.magnetics.MagneticConfig.getComputed(...) instead, which uses the fusionsc field
	convention of [grid.nPhi, grid.nZ, grid.nR, 3].
	
	Obtains the magnetic field of a configuration on a specific grid.
	
	Parameters:
		- config: Description of the magnetic field
		- grid: Grid to compute the field on.
	
	Returns:
		An array of shape [3, grid.nPhi, grid.nZ, grid.nR] containing the magnetic field.
		The directions along the first coordinate are phi, z, r.
	"""
	import warnings
	warnings.warn(DeprecationWarning(
		"fusionsc.flt.fieldValues(...) is deprecated and will be removed in version 3.0. Use fusionsc.magnetics.MagneticConfig.getComputed(...) instead."
	))
	
	import numpy as np
	
	field = await config.compute.asnc(grid)
	fieldData = await data.download.asnc(field.data.computedField.data)
	
	return np.asarray(fieldData).transpose([3, 0, 1, 2])

@asyncFunction
async def poincareInPhiPlanes(points, config, phiPlanes, turnLimit, **kwArgs):
	"""
	Computes the Poincaré map starting from a given set of points on a config.
	
	Mostly equivalent to :code:`trace(points, config, phiPlanes = phiPlanes, turnLimit = turnLimit, **kwArgs)["poincareHits"]`.
	
	Parameters:
		- points: Starting points for the trace. Can be any shape, but the first dimension must have a size of 3 (x, y, z).
		- phiPlanes: 1D list of intersection plane angles (in radian)
		- config: Magnetic configuration. If this is not yet computed, you also need to specify the 'grid' parameter (see the documentation of trace).
	
	Returns:
		An array of shape [5, len(phiPlanes)] + points.shape[1:] + [turnLimit].
		The entries in dimension 1 are [x, y, z, forward connection length, backward connection length].
		If the forward- or backward-going field lines from this point hit no geometry, the returned length will be negative
		and its absolute magnitude will indicate the remaining field line length in that direction.
	
	For the extra arguments, see the parameters to trace().
	"""
	result = await trace.asnc(points, config, turnLimit = turnLimit, phiPlanes = phiPlanes, **kwArgs)
	return result["poincareHits"]

@asyncFunction
async def connectionLength(points, config, geometry, **kwargs):
	"""
	Computes the connection-length of the given points in a certain geometry.
	
	Mostly equivalent to :code:`trace(points, config, geometry = geometry, collisionLimit = 1, **kwargs)["endPoints"][3]`.
	
	Parameters:
		- points: Starting points for the trace. Can be any shape, but the first dimension must have a size of 3 (x, y, z).
		- config: Magnetic configuration. If this is not yet computed, you also need to specify the 'grid' parameter.
		- geometry: Device geometry. If this is not yet indexed, you need to specify the 'geometryGrid' parameter (see the documentation of trace).

	Returns:
		An array of shape `points.shape[1:]` indicating the forward connection length of the given point.
	""" 
	result = await trace.asnc(points, config, geometry = geometry, collisionLimit = 1, resultFormat = 'raw', **kwargs)
	endPoints = np.asarray(result.endPoints)
	return endPoints[3]

@asyncFunction
async def followFieldlines(points, config, recordEvery = 1, **kwargs):
	"""
	Follows magnetic field lines.
	
	Mostly equivalent to :code:`(lambda x: return x["fieldLines"], x["fieldStrengths"])(trace(points, config, recordEvery, **kwargs))`.
	
	Parameters:
		- points: Starting points for the trace. Can be any shape, but the first dimension must have a size of 3 (x, y, z).
		- config: Magnetic configuration. If this is not yet computed, you also need to specify the 'grid' parameter.
		- recordEvery: Number of tracing steps between each recorded point.

	Returns:
		A tuple holding:
		- An array of shape `points.shape     + [max. field line length]` indicating the field line point locations
		- An array of shape `points.shape[1:] + [max. field line length]` indicating the field strength at those points
	""" 
	result = await trace.asnc(points, config, recordEvery = recordEvery, **kwargs)
	return result["fieldLines"], result["fieldStrengths"]
	
@asyncFunction
async def trace(
	points, config,
	geometry = None,
	grid = None, geometryGrid = None,
	mapping = None,
	
	# Limits to stop tracing
	distanceLimit = 1e4, turnLimit = 0, stepLimit = 0, stepSize = 1e-3, collisionLimit = 0,
	
	# Plane intersections
	phiPlanes = [],
	
	# Diffusive transport specification
	isotropicDiffusionCoefficient = None, rzDiffusionCoefficient = None,
	parallelConvectionVelocity = None, parallelDiffusionCoefficient = None,
	meanFreePath = 1, meanFreePathGrowth = 0,
	
	# Direction change
	direction = "forward",
	
	# Direction recording
	recordEvery = 0,
	
	# Return format
	resultFormat = 'dict',
	
	# Adaptive step size control
	targetError = None, relativeErrorTolerance = 1, minStepSize = 0, maxStepSize = 0.2,
	errorEstimationDistance = None,
	
	# Minimum tracing before processing collisions
	ignoreCollisionsBefore = 0,
	
	# Whether field line reversal is allowed
	allowReversal = False,
	
	# Record mode for plane hits ("auto", "lastInTurn" or "everyHit")
	planeRecordMode = "auto"
):
	"""
	Performs a tracing request.
	
	Parameters:
		- points: Starting points. Array-like of shape [3, ...] (must be at least 1D, first dimension is x, y, z).
		- config (magnetics.MagneticConfig): Magnetic configuration. If this is not yet computed, you must specify the 'grid' parameter.
		- geometry (geometry.Geometry): Geometry to use for intersection tests of field lines. If this is not yet indexed, you must specify the 'geometryGrid' parameters.
		- grid (service.ToroidalGrid.Reader or service.ToroidalGrid.Builder): Toroidal grid to compute the magnetic field on prior to tracing.
		- geometryGrid (service.CartesiaGrid.Reader or service.CartesianGrid.Builder): Cartesian grid to index the geometry triangles on (used to accelerate intersection computations).
	
		- distanceLimit: Maximum field line tracing length. 0 or negative interpreted as infinity.
		- turnLimit: Maximum number of device turn to trace field line for. 0 or negative interpreted as infinity.
		- stepLimit: Maximum number of steps to trace for. 0 interpreted as infinity. May not be negative.
		- stepSize: Step size for each tracing step (in meters).
		- collisionLimit: Maximum number of collisions a field line may perform (e.g. 1 = termination at first collision. 2 at second collision etc.). Must not be negative.
		  0 interpreted as infinity.
		  
		- phiPlanes (list): List of planes to perform intersections with. Objects can be either phi values in radians, or objects castable
		  to service.Plane (readers / builders for this type, or YAML strings, or dictionaries filled with the appropriate structure). 
	
		- isotropicDiffusionCoefficient: If set, enables diffusive tracing and specifies the isotropic / perpendicular diffusion coefficient to use in the
		  underlying diffusive tracing model. If set, either parallelConvectionVelocity or parallelDiffusionCoefficient must also be specified.
		- rzDiffusionCoefficient: Similar to isotropicDiffusionCoefficient, but displacements are only done in the rz plane
		- parallelConvectionVelocity: Parallel streaming velocity to assume for a single-directional diffusive tracing model.
		- parallelDiffusionCoefficient: Parallel diffusion coefficient to assume for a fully bidireactional doubly-diffusive tracing model.
	
		- meanFreePath: Mean (not fixed!) free path to use to sample the tracing distance between diffusive displacement steps.
		- meanFreePathGrowth: Amount by which to increase the mean free path after every displacement step. This parameter prevents
		  extreme growths of computing costs at low perpendicular diffusion coefficients.
	
		- mapping: Field line mapping to accelerate tracing.
		
		- direction: Indicates the tracing direction. One of "forward" (field direction), "backward" (against field), "cw" (clockwise), or "ccw" (counter-clockwise)
		
		- recordEvery: When set to > 0, instructs the tracer to record the fieldline every "recordEvery"-th step.
		
		- targetError: Step size to adjust steps towards.
		- relativeErrorTolerance: If the error estimator indicates that the error is more than targetError * (1 + relativeErrorTolerance), the step will be repeated
		  with adjusted step size. Otherwise, the adjusted step size will be used for the next step.
		- minStepSize: Minimum step size for adaptive controller
		- maxStepSize: Maximum step size for adaptive controller
		
		- errorEstimationDistance: Maximum trace length to be assumed for the purpose of error estimation. If this is not set, the service will try to estimate it
		  from the limits. Can be set to "step" to indicate per-step error targets.
		
		- ignoreCollisionsBefore: Minimum distance to trace before collisions will be actively processed. Useful when starting a trace on / very close to a mesh and
		  not wanting to immediately have this mesh terminate the trace.
		
		- allowReversal: If this is set to false (default), a reversal of toroidal magnetic field orientation along the field line will terminate the
		  tracing process. This prevents long traces on field lines encircling coils.
		
		- planeRecordMode: Recording style to use for Poincare hits. Possible values:
		  - "auto": Uses "lastInTurn" when only using phi-planes, and "everyHit" with a one-time warning notification otherwise.
		  - "lastInTurn": Indexes the hits per turn while "everyHit" records all hits of the plane.
		  - "everyHit": Returns all hits in order of trace direction of field line.
	
	Returns:
		The format of the result depends on the `resultFormat` parameter.
		
		-	If `resultFormat == 'dict'`:
		
			A dictionary holding the following entries (more can be added in future versions)
			
			- *endPoints*: A numpy array of shape `[4] + startPoints.shape[1:]`. The first 3 components are the x, y, and z positions of
			  the field lines' end points. The 4th component is the total length of the field line.
			- *poincareHits*: A numpy array of shape `[5, len(phiPlanes)] + startPoints.shape[1:] + [maxTurns]` with maxTurns being a number <=
			  turnLimit indicating the maximum turn count of any field line, or - if planeRecordMode is "everyHit" - arbitrary. The first 3 components
			  of the first dimension are the x, y, and z coordinates of the phi plane intersections (commonly used for Poincaré maps).
			  The next two components indicate the forward and backward connection lengths respectively to the next geometry collision along the
			  field line. If the field line ends in that direction without a collision (e.g. closed field line, or no geometry specified), a negative
			  number is returned whose absolute value corresponds to the remaining length in that direction. Non-existing points (due to field lines
			  not all having same turn / hit counts) have their values set to NaN.
			- *stopReasons*: A numpy array of shape `startPoints.shape[1:]` that indicates for each point the final reason why the trace was stopped.
			  The dtype of the array is fusionsc.service.FLTStopReason.
			- *fieldLines*: A numpy array of shape `startPoints.shape + [max. field line length]` containing steps recorded at specified intervals
			  (see parameter `recordEvery. Padded with NaN.
			- *fieldStrengths*: A numpy array of shape `startPoints.shape[1:] + [max. field line length]` holding the absolute magnetic field
			  strength at the recorded field line points. Padded with 0.
			- *endTags*: A dict containing a numpy array of type fusionsc.service.TagValue for each tag name present in the geometry. Each array is
			  of shape `startPoints.shape[1:]`, and its values indicate the tags associated with the final geometry hit. This gives information
			  about the meshes impacted by the field lines.
			- *responseSize*: The total size of the response size in bytes (mainly for profiling purposes).
		
		-	If `resultFormat == 'raw'`:
			
			An instance of fusionsc.service.FLTResponse.Reader (for more efficient storage and later decoding).
	"""
	
	if stepSize < 0.05 and mapping is not None:
		import warnings
		warnings.warn(
"""Note: You are using a mapping, but still have a rather small step size. For better performance, you might consider increasing the step size to
a larger value so that you can take advantage of the stable long-range calculation (your step size limit should be defined by the required accuracy
for geometry intersection tests, the magnetic field tracing accuracy should not degrade at large steps when using a field line mapping"""
		)
		
	assert parallelConvectionVelocity is None or parallelDiffusionCoefficient is None
	if isotropicDiffusionCoefficient is not None or rzDiffusionCoefficient is not None: 
		assert parallelConvectionVelocity is not None or parallelDiffusionCoefficient is not None
	
	assert resultFormat in ['dict', 'raw'], "resultFormat parameter must be 'dict' or 'raw'"
	
	config = await config.compute.asnc(grid)
	computedField = config.data.computedField
	
	if geometry is not None:	
		geometry = await geometry.index.asnc(geometryGrid)
		indexedGeometry = geometry.data.indexed
	
	request = service.FLTRequest.newMessage()
	request.startPoints = points
	request.field = computedField
	request.stepSize = stepSize
	
	request.distanceLimit = distanceLimit
	request.stepLimit = stepLimit
	request.collisionLimit = collisionLimit
	request.turnLimit = turnLimit
	
	request.allowReversal = allowReversal
	
	request.ignoreCollisionsBefore = ignoreCollisionsBefore
	
	assert direction in ["forward", "backward", "cw", "ccw"]
	
	if direction == "field":
		request.forward = True
	
	if direction == "backward":
		request.forward = False
	
	if direction == "cw":
		request.forwardDirection = "ccw"
		request.forward = False
	
	if direction == "ccw":
		request.forwardDirection = "ccw"
		request.forward = True
			
	# Diffusive transport model
	if isotropicDiffusionCoefficient is not None or rzDiffusionCoefficient is not None:
		assert isotropicDiffusionCoefficient is None or rzDiffusionCoefficient is None, "Only one of isotropic or rz diffusion can be specified"
		
		if isotropicDiffusionCoefficient is not None:
			request.perpendicularModel.isotropicDiffusionCoefficient = isotropicDiffusionCoefficient
		else:
			request.perpendicularModel.rzDiffusionCoefficient = rzDiffusionCoefficient
		
		request.parallelModel.meanFreePath = meanFreePath
		request.parallelModel.meanFreePathGrowth = meanFreePathGrowth
		
		if parallelConvectionVelocity is not None:
			request.parallelModel.convectiveVelocity = parallelConvectionVelocity
		else:
			request.parallelModel.diffusionCoefficient = parallelDiffusionCoefficient
	
	# Poincare maps
	if len(phiPlanes) > 0:
		def makePlane(x):
			if isinstance(x, numbers.Number):
				return {"orientation" : {"phi" : x}}
			
			return x
			
		request.planes = [
			makePlane(x)
			for x in phiPlanes
		]
	
	# Adjust planeRecordMode 'auto' argument
	if planeRecordMode == "auto":
		planeRecordMode = "lastInTurn"
		
		for plane in request.planes:
			if plane.orientation.which_() != "phi":
				planeRecordMode = "everyHit"
				
				import warnings
				warnings.warn("Using a plane that is not a phi-plane switches the default intersection "
				"recording (from 'lastPerTurn' to 'every hit'), which means that the last axis no longer "
				"indicates the turn count. Manually set the 'planeRecordMode' argument to one of these "
				"values to suppress this warning")
				
				break
	
	request.planeIntersectionRecordMode = planeRecordMode
	
	# Field line following
	if recordEvery > 0:
		request.recordEvery = recordEvery
	
	if geometry is not None:
		request.geometry = indexedGeometry
	
	if mapping is not None:
		if isinstance(mapping, wrappers.RefWrapper):
			request.mapping = mapping.ref
		elif isinstance(mapping, MappingWithGeometry):
			request.geometryMapping = mapping.data
		else:
			raise ValueError("Invalid type of mapping")
	
	if targetError is not None:		
		# Try to estimate error estimation distance
		if errorEstimationDistance is None and distanceLimit > 0:
			errorEstimationDistance = distanceLimit
		
		if errorEstimationDistance is None and turnLimit > 0:
			# Add a leniency factor of 2 for the turn-based distance estimation
			errorEstimationDistance = 2 * 2 * np.pi * computedField.grid.rMax
		
		if errorEstimationDistance is None and stepLimit > 0:
			errorEstimationDistance = "step"
			targetError /= stepLimit
		
		assert errorEstimationDistance is not None, "Adaptive error was specified" \
			" but no effective distance limit could be inferred. Please specify" \
			" the errorEstimationDistance parameter. If you want to target per-" \
			"step errors, please set errorEstimationDistance to 'step'"
					
		adaptive = request.stepSizeControl.initAdaptive()
		adaptive.targetError = targetError
		adaptive.relativeTolerance = relativeErrorTolerance
		adaptive.min = minStepSize
		adaptive.max = maxStepSize
		
		if errorEstimationDistance != "step":
			adaptive.errorUnit.integratedOver = errorEstimationDistance
	
	# Perform the tracing
	response = await _tracer().trace(request)
	
	# Decode the response
	return decodeTraceResponse(response, resultFormat)

def decodeTraceResponse(response: service.FLTResponse.ReaderOrBuilder, resultFormat: str = 'dict'):
	"""
	Decodes an FLT response according to the requested format.
	
	Parameters:
		- response: A raw field line tracer response.
		- resultFormat: Either "raw" or "dict".
	
	Returns (depending on resultFormat parameter):
		- 'raw': An instance of fusionsc.service.FLTResponse.ReaderOrBuilder containing the raw
		  response of the field line tracer (basically the input arguments).
		- 'dict': A dict of numpy arrays containing pythonic view on the response data. See the
		  documentation of trace() for the full contents.
	"""
	if resultFormat == 'raw':
		return response
	
	endTags = {
		str(tagName) : tagData
		for tagName, tagData in zip(response.tagNames, np.asarray(response.endTags))
	}
	
	result = {
		"endPoints" : np.asarray(response.endPoints),
		"poincareHits" : np.asarray(response.poincareHits),
		"stopReasons" : np.asarray(response.stopReasons),
		"fieldLines" : np.asarray(response.fieldLines),
		"fieldStrengths" : np.asarray(response.fieldStrengths),
		"endTags" : endTags,
		"numSteps" : np.asarray(response.numSteps),
		"responseSize" : response.totalBytes_()
	}
	return result

@asyncFunction
async def findAxis(
	field, grid = None, startPoint = None,
	stepSize = 0.001, nTurns = 10, nIterations = 10, nPhi = 200, direction = "ccw", mapping = None,
	targetError = None, relativeErrorTolerance = 1, minStepSize = 0, maxStepSize = 0.2,
	islandM = 1
):
	"""
	Computes the magnetic axis by repeatedly tracing a Poincaré map and averaging the points.
	
	Parameters:
		- field: The magnetic field.
		- grid: Optional grid to compute the magnetic field on. If not specified, the field must
		  already be computed.
		- startPoint: Start point for the magnetic axis search. If unspecified, defaults to the
		  midpoint of the phi = 0 cross section.
		- stepSize: Step size for tracing of (if using the adaptive error estimator) for the inital
		  step size guess.
		- nTurns: Number of points to average per iteration for the magnetic axis computation.
		- nIterations: Number of axis search iterations.
		- nPhi: Number of phi points to calculate the final magnetic axis trace on.
		- direction: Direction to calculate the axis in.
		- mapping: Field line mapping to use for accelerating the trace.
		- targetError: Enables adaptive stepping by specifying a maximum error tolerance.
		- relativeErrorTolerance, minStepSize, maxStepSize: See trace() for more information.
		- islandM: When set, this routine will identify the O cycle of the island the starting
		  point is in (which can still be the axis).
	
	Returns:
		A tuple holding the xyz-position of the axis starting point and a numpy array holding the field line
		corresponding to the magnetic axis.
	"""
	assert direction in ["cw", "ccw", "co-field", "counter-field"]
	
	field = await field.compute.asnc(grid)
	computed = field.data.computedField
	
	# If start point is not provided, use grid center
	if startPoint is None:
		fieldGrid = computed.grid
		
		startPoint = [0.5 * (fieldGrid.rMax + fieldGrid.rMin),
			0, 0.5 * (fieldGrid.zMax + fieldGrid.zMin)
		]
	
	startPoint = np.asarray(startPoint)
	
	# The FLT service object has 2 methods for axis location. The original "findAxis"
	# method can only serve a single start point. In the past, this would be circumvented
	# by averaging the start points together. This was unintuitive behavior.
	# Therefore, a new method "findAxisBatch" was implemented that would perform axis
	# search for multiple start points (without requiring a single function call per start
	# point and allowing direct return of all axis traces as tensor).
	#
	# For compatibility with old servers, in case only 1 point is needed the call is
	# routed through the old function call (batch == False).
	
	batch = False
	if len(startPoint.shape) > 1:
		batch = True
	
	request = service.FindAxisRequest.newMessage()
	request.field = computed
	request.stepSize = stepSize
	request.nTurns = nTurns
	request.nIterations = nIterations
	request.nPhi = nPhi
	request.islandM = islandM
	
	if not batch:
		request.startPoint = startPoint
	
	if mapping is not None:
		if isinstance(mapping, wrappers.RefWrapper):
			request.mapping = mapping.ref
		elif isinstance(mapping, MappingWithGeometry):
			request.geometryMapping = mapping.data
		else:
			raise ValueError("Invalid type of mapping")
	
	if targetError is not None:		
		adaptive = request.stepSizeControl.initAdaptive()
		adaptive.targetError = targetError
		adaptive.relativeTolerance = relativeErrorTolerance
		adaptive.min = minStepSize
		adaptive.max = maxStepSize
		adaptive.errorUnit.integratedOver = 2 * np.pi * np.amax(np.sqrt(startPoint[0]**2 + startPoint[1]**2)) * nTurns
	
	if batch:
		response = await _tracer().findAxisBatch(startPoint, request)
	else:
		response = await _tracer().findAxis(request)
	
	axis = np.asarray(response.axis)
	x, y, z = axis
	phiVals = np.arctan2(y, x)
	dPhi = phiVals[...,1] - phiVals[...,0]
	dPhi = ((dPhi + np.pi) % (2 * np.pi)) - np.pi
	dPhi = np.mean(dPhi)
	
	swap = False
	
	if dPhi > 0 and direction == "cw":
		swap = True
	
	if dPhi < 0 and direction == "ccw":
		swap = True
	
	if direction == "counter-field":
		swap = True
	
	if swap:
		axis = axis[..., ::-1]
	
	return np.asarray(response.pos), axis

@asyncFunction
async def findLCFS(
	field, geometry, p1, p2,
	grid = None, geometryGrid = None, stepSize = 0.01, tolerance = 0.001, nScan = 8, distanceLimit = 3e3, mapping = None,
	targetError = None, relativeErrorTolerance = 1, minStepSize = 0, maxStepSize = 0.2
):
	"""
	Compute the position of the last closed flux surface
	"""
	field = await field.compute.asnc(grid)
	computedField = field.data.computedField
	
	geometry = await geometry.index.asnc(geometryGrid)
	indexedGeometry = geometry.data.indexed
	
	request = service.FindLcfsRequest.newMessage()
	request.p1 = p1
	request.p2 = p2
	request.stepSize = stepSize
	request.tolerance = tolerance
	request.field = computedField
	request.geometry = indexedGeometry
	request.nScan = nScan
	request.distanceLimit = distanceLimit
	
	if targetError is not None:		
		adaptive = request.stepSizeControl.initAdaptive()
		adaptive.targetError = targetError
		adaptive.relativeTolerance = relativeErrorTolerance
		adaptive.min = minStepSize
		adaptive.max = maxStepSize
		adaptive.errorUnit.integratedOver = distanceLimit
	
	if mapping is not None:
		if isinstance(mapping, wrappers.RefWrapper):
			request.mapping = mapping.ref
		elif isinstance(mapping, MappingWithGeometry):
			request.geometryMapping = mapping.data
		else:
			raise ValueError("Invalid type of mapping")
	
	response = await _tracer().findLcfs(request)
	
	return np.asarray(response.pos)

@asyncFunction
async def axisCurrent(
	field, current,
	grid = None, startPoint = None, stepSize = 0.001, nTurns = 10, nIterations = 10, nPhi = 200, direction = None,
	mapping = None,
	targetError = None, relativeErrorTolerance = 1, minStepSize = 0, maxStepSize = 0.2
):
	"""
	Configuration that places a current on the axis of a given magnetic configuration.
	
	This function computes the magnetic axis of a magnetic configuration, interprets it as a coil, and
	creates a magnetic configuration with a given current specified on this axis.
	
	Parameters:
		field: Magnetic field configuration
		current: On-axis current to apply
		grid: Grid to use for magnetic field calculation if the field is not yet computed
	
	Returns:
		The magnetic configuration corresponding to the on-axis current's field.
	"""
	
	assert field.data.which_() == "computedField" or grid is not None, "The magnetic field must either be precomputed or a grid must be specified."
	
	if direction is None:
		direction = "ccw"
		
		import warnings
		warnings.warn(
""" !!! No direction convention specified, default unsuitable for W7-X !!!

You have specified no direction convention for the on-axis plasma current. By default, the fusionsc.flt.axisCurrent
assumes a counter-clockwise current (mathematically positive). The W7-X convention specifies a clock-wise current.
Please use the argument 'direction = "ccw"' (if you are sure you want a counter-clockwise current) or use the
fusionsc.devices.w7x.axisCurrent function instead (which also takes care of the starting point)."""
		)
	
	if startPoint is None:		
		import warnings
		warnings.warn(
""" !!! No starting point specified, default unsuitable for W7-X !!!

You have specified no start point for the magnetic axis search. The default value (which starts at the grid center at phi == 0)
is poorly suited for W7-X. Consider using 'startPoint = [6, 0, 0]' or using fusionsc.devices.w7x.axisCurrent, which has W7-X-tailored
default values."""
		)
	
	
	result = await findAxis.asnc(field, grid, startPoint, stepSize, nTurns, nIterations, nPhi, direction, mapping)
	_, axis = result
	
	result = magnetics.MagneticConfig()
	
	filField = result.data.initFilamentField()
	filField.current = current
	filField.biotSavartSettings.stepSize = stepSize
	filField.filament.inline = axis.T
	
	return result

@asyncFunction
async def computeMapping(
	field, mappingPlanes, r, z,
	grid = None, distanceLimit = 1e3, padding = 2, numPlanes = 20, stepSize = 0.001,
	u0 = [0.5], v0 = [0.5],
	toroidalSymmetry = None,
	
	targetError = None, relativeErrorTolerance = 1, minStepSize = 0, maxStepSize = 0.2,
	errorEstimationDistance = None,
):
	"""
	Computes a reversible field line mapping. This mapping type divides the device into toroidal
	sections. Each section is covered by a curved conforming hexahedral grid. The "mapping planes"
	define the toroidal sections at which the different sections meet. Within a sections, field lines
	can be interpolated in the r-z plane using phi-independent u,v coordinates on the grid. When crossing
	into other sections, the field line mapping must be inverted to construct new grid coordinates.
	
	The tracing of the sections is started from planes lying in-between the mapping planes.
	
	Parameters:
		- field: A magnetic field to create the mapping over.
		- mappingPlanes: An array-like of radial angles. Must be in counterclockwise order (but can start and stop anywhere)
		- r: Major radius values of the starting points to trace the grid from
		- z: Z coordinate values of the starting points to trace the grid from
		- grid: Grid to use for field computation (if field is not computed yet)
		- distanceLimit: Maximum field line length to trace. Can often be reduced aggressively.
		- padding: Number of planes to add beyond the mapping planes on each section. Currently must be at least 2.
		- numPlanes: Number of planes to record for each half-section trace.
		- stepSize: Step size to use for tracing.
		- u0 (number or sequence of numbers, ranging from 0 to 1): Per-section starting grid coordinates for mapping, radial component
		- v0 (number or sequence of numbers, ranging from 0 to 1): Per-section starting grid coordinates for mapping, vertical component
	
	Returns:
		A DataRef pointing to a to-be-initialized field line mapping.
	"""
	import numbers
	
	field = await field.compute.asnc(grid)
	computedField = field.data.computedField
	
	backend = backends.activeBackend()
	
	if toroidalSymmetry is None:
		toroidalSymmetry = computedField.grid.nSym
	
	request = service.RFLMRequest.newMessage()
	request.gridR = r
	request.gridZ = z
	request.mappingPlanes = mappingPlanes
	request.field = computedField
	request.numPlanes = numPlanes
	request.numPaddingPlanes = padding
	request.distanceLimit = distanceLimit
	request.stepSize = stepSize
	request.nSym = toroidalSymmetry
	
	if targetError is not None:
		assert errorEstimationDistance is not None, "Can only use adaptive integration if a characteristic distance for error estimation is set. This is important because the accuracy of the mapping will carry over into its application."
		
		adaptive = request.stepSizeControl.initAdaptive()
		adaptive.min = minStepSize
		adaptive.max = maxStepSize
		adaptive.targetError = targetError
		
		if errorEstimationDistance == "step":
			adaptive.errorUnit = "step"
		else:
			adaptive.errorUnit.integratedOver = errorEstimationDistance
	
	request.u0 = [u0] if isinstance(u0, numbers.Number) else u0
	request.v0 = [v0] if isinstance(v0, numbers.Number) else v0
	
	return FieldlineMapping(_mapper().computeRFLM(request).pipeline.mapping)

@asyncFunction
async def calculateIota(
	field, startPoints, turnCount, grid = None, 
	axis = None, unwrapEvery = 1,
	distanceLimit = 1e4, 
	targetError = 1e-4, relativeErrorTolerance = 1, minStepSize = 0, maxStepSize = 0.2,
	islandM = 1,
	mapping = None
):
	# Make sure we have a computed field reference
	field = await field.compute.asnc(grid)
	
	startPoints = np.asarray(startPoints)
	
	# Determine axis shape (required for phase unwrapping)
	if axis is None:
		# startPoint = startPoints.reshape([3, -1]).mean(axis = 1)
		p, axis = await findAxis.asnc(field, nTurns = 10 * islandM, startPoint = startPoints, islandM = islandM, targetError = targetError, relativeErrorTolerance = relativeErrorTolerance, minStepSize = minStepSize, maxStepSize = maxStepSize)
	
	xAx, yAx, zAx = axis
	rAx = np.sqrt(xAx**2 + yAx**2)
	
	# Initialize trace request
	request = service.FLTRequest.newMessage()
	request.stepSize = 0.001
	request.startPoints = startPoints
	request.field = field.data.computedField
	request.distanceLimit = distanceLimit
	request.turnLimit = turnCount
	
	# Set mapping
	if mapping is not None:
		import warnings
		warnings.warn("Using a mapping in iota and surface calculations is currently experimental." +
			" Do not use long connection lengths as this will invalidate the Fourier calculations"
		)
		
		if isinstance(mapping, wrappers.RefWrapper):
			request.mapping = mapping.ref
		elif isinstance(mapping, MappingWithGeometry):
			request.geometryMapping = mapping.data
		else:
			raise ValueError("Invalid type of mapping")
	
	calcIota = request.fieldLineAnalysis.initCalculateIota()
	calcIota.unwrapEvery = unwrapEvery
	calcIota.islandM = islandM
	
	if len(rAx.shape) == 1:
		calcIota.axis.shared = {
			'r' : rAx,
			'z' : zAx
		}
	else:
		calcIota.axis.individual = {
			'r' : rAx,
			'z' : zAx
		}
	
	if mapping is None:
		adaptive = request.stepSizeControl.initAdaptive()
		adaptive.targetError = targetError
		adaptive.relativeTolerance = relativeErrorTolerance
		adaptive.min = minStepSize
		adaptive.max = maxStepSize
		
	errorEstimationDistance = min(turnCount * 2 * np.pi * field.data.computedField.grid.rMax, distanceLimit)
	adaptive.errorUnit.integratedOver = errorEstimationDistance
	
	# Perform trace command
	response = await _tracer().trace(request)
	return np.asarray(response.fieldLineAnalysis.iotas)

@asyncFunction
async def calculateFourierModes(
	field, startPoints, turnCount,
	nMax = 1, mMax = 1, toroidalSymmetry = 1, aliasThreshold = None,
	grid = None, stellaratorSymmetric = False,
	unwrapEvery = 1, recordEvery = 10, distanceLimit = 0,
	targetError = 1e-4, relativeErrorTolerance = 1, minStepSize = 0, maxStepSize = 0.2,
	islandM = 1,
	mapping = None
):
	if aliasThreshold is None:
		aliasThreshold = 0.05 / turnCount
		
	field = await field.compute.asnc(grid)
	
	startPoints = np.asarray(startPoints)
	
	# Calculate iota
	iotas = await calculateIota.asnc(
		field, startPoints, turnCount * 20,
		grid = None, axis = None, unwrapEvery = unwrapEvery,
		distanceLimit = distanceLimit,
		targetError = targetError, relativeErrorTolerance = relativeErrorTolerance, minStepSize = minStepSize, maxStepSize = maxStepSize,
		islandM = islandM,
		mapping = mapping
	)
	
	# Initialize Fourier trace request
	request = service.FLTRequest.newMessage()
	request.stepSize = 0.001
	request.startPoints = startPoints
	request.field = field.data.computedField
	request.distanceLimit = distanceLimit
	request.turnLimit = turnCount
	
	# Set mapping
	if mapping is not None:
		if isinstance(mapping, wrappers.RefWrapper):
			request.mapping = mapping.ref
		elif isinstance(mapping, MappingWithGeometry):
			request.geometryMapping = mapping.data
		else:
			raise ValueError("Invalid type of mapping")
	
	calcSurf = request.fieldLineAnalysis.initCalculateFourierModes()
	calcSurf.iota = iotas
	calcSurf.nMax = nMax
	calcSurf.mMax = mMax
	calcSurf.recordEvery = recordEvery
	calcSurf.toroidalSymmetry = toroidalSymmetry
	calcSurf.modeAliasingThreshold = aliasThreshold 
	calcSurf.stellaratorSymmetric = stellaratorSymmetric
	calcSurf.islandM = islandM
	
	if mapping is None:
		adaptive = request.stepSizeControl.initAdaptive()
		adaptive.targetError = targetError
		adaptive.relativeTolerance = relativeErrorTolerance
		adaptive.min = minStepSize
		adaptive.max = maxStepSize
	
	if distanceLimit > 0:
		errorEstimationDistance = min(turnCount * 2 * np.pi * field.data.computedField.grid.rMax, distanceLimit)
	else:
		errorEstimationDistance = turnCount * 2 * np.pi * field.data.computedField.grid.rMax
	
	adaptive.errorUnit.integratedOver = errorEstimationDistance
	
	# Perform trace command
	response = await _tracer().trace(request)
	modes = response.fieldLineAnalysis.fourierModes
	
	result = {
		"surfaces" : magnetics.SurfaceArray(modes.surfaces),
		
		"iota" : iotas,
		"theta" : np.asarray(modes.theta0),
		"rCos" : np.asarray(modes.surfaces.rCos),
		"zSin" : np.asarray(modes.surfaces.zSin),
		"mPol" : np.asarray(modes.mPol)[:,None],
		"nTor" : np.asarray(modes.nTor)[None,:]
	}
	
	if modes.surfaces.which_() == "nonSymmetric":
		result["rSin"] = np.asarray(modes.surfaces.nonSymmetric.rSin)
		result["zCos"] = np.asarray(modes.surfaces.nonSymmetric.zCos)
	
	return result
	
	
