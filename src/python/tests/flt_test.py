import fusionsc as fsc
from fusionsc.devices import jtext

import pytest
import numpy as np

@pytest.fixture(scope="session")
def grid():
	result = jtext.defaultGrid()
	result.nR = 32
	result.nZ = 32
	result.nPhi = 1
	return result

@pytest.fixture(scope="session")
def field(grid):
	geqdsk = jtext.exampleGeqdsk()
	field = fsc.magnetics.MagneticConfig.fromEFit(geqdsk)
	return field.compute(grid)

@pytest.fixture(scope="session")
def geometry():
	return jtext.pfcs(0.24).index(jtext.defaultGeometryGrid())

@pytest.fixture(scope="session")
def upstreamPoints(field, geometry):
	# Calculate LCFS position
	lcfsPosition = fsc.flt.findLCFS(field, geometry, [1.05, 0, 0], [1.4, 0, 0], tolerance = 1e-3, targetError = 1e-3)
	lcfsPosition = lcfsPosition - [0.01, 0, 0]
	
	# Poincare planes
	planes = np.linspace(0, 2 * np.pi, 5, endpoint = False)
	points = fsc.flt.poincareInPhiPlanes(lcfsPosition, field, planes, 5, geometry = geometry, targetError = 1e-3)
	
	return points[:3]

@pytest.fixture(scope="session")
def heatCam():
	proj = fsc.hfcam.toroidalProjection(
		w = 400, h = 300,
		phi = np.radians(130), rTarget = 0.8, zTarget = 0,
		verticalInclination = np.radians(20), horizontalInclination = np.radians(0),
		distance = 0.3,
		viewportHeight = 0.4, fieldOfView = np.radians(30)
	)
	cam = fsc.hfcam.make(
		proj,
		jtext.hfsLimiter() + jtext.target() + jtext.lfsLimiter(0.24) + jtext.topLimiter(0.24) + jtext.bottomLimiter(0.24)
	)
	return cam

@pytest.mark.parametrize("recMode", ['lastInTurn', 'everyHit'])
def test_poincare(field, geometry, recMode):
	rStart = np.linspace(1.05, 1.1, 10)
	zStart = 0 * rStart
	yStart = 0 * rStart
	
	pcPlanes = [0, np.pi, "orientation: {normal: [1, 0, 0]}"]
	
	result = fsc.flt.poincareInPhiPlanes([rStart, yStart, zStart], field, pcPlanes, 200, geometry = geometry, targetError = 1e-3, planeRecordMode = recMode)

def test_axis(field):
	fsc.flt.findAxis(field, startPoint = [1.1, 0, 0], targetError = 1e-3)

def test_fieldline_diffusion(field, geometry, upstreamPoints, tmp_path):
	traceResult = fsc.flt.trace(
		upstreamPoints, field, geometry,
		distanceLimit = 1e4, stepSize = 1e-3, collisionLimit = 1,
		
		rzDiffusionCoefficient = 1,
		parallelConvectionVelocity = 1e5,
		
		meanFreePath = 5, meanFreePathGrowth = 0.1,
		
		targetError = 1e-2
	)
	
	fsc.export.exportTrace(traceResult, str(tmp_path / "test.nc"))
	fsc.export.exportTrace(traceResult, str(tmp_path / "test.mat"))
	fsc.export.exportTrace(traceResult, str(tmp_path / "test.json"))

def test_anisotropic_diffusion(field, geometry, upstreamPoints, heatCam):
	traceResult = fsc.flt.trace(
		upstreamPoints, field, geometry,
		distanceLimit = 1e4, stepSize = 1e-3, collisionLimit = 1,
		
		isotropicDiffusionCoefficient = 1,
		parallelDiffusionCoefficient = 1e6,
		
		meanFreePath = 5, meanFreePathGrowth = 0.1,
		
		targetError = 1e-2
	)
	
	# Analyze
	cam = heatCam.clone()
	cam.addPoints(traceResult['endPoints'][:3], r = 0.01)
	cam.get()
	
	# Check symmetrize function
	fsc.flt.symmetrize(traceResult['endPoints'][:3], 10, True)

def test_mapping1(field):
	mapping = fsc.flt.computeMapping(
		field,
		np.radians([0, 36]),
		np.linspace(0.8, 1.2, 5), np.linspace(-0.2, 0.2, 5),
		distanceLimit = 2 * np.pi * 1.5 + 1,
		toroidalSymmetry = 5
	)
	fsc.asnc.wait(mapping)
	
def test_mapping2(field):
	mapping = fsc.flt.computeMapping(
		field,
		np.radians([0, 180]),
		np.linspace(0.8, 1.2, 5), np.linspace(-0.2, 0.2, 5),
		distanceLimit = 2 * np.pi * 1.5 + 1,
		toroidalSymmetry = 1
	)
	fsc.asnc.wait(mapping)

def test_iota(field):
	fsc.flt.calculateIota(field, [1.1, 0, 0], 200, targetError = 1e-3)
	
def test_surf(field):
	points = np.linspace([1.1, 0, 0], [1.2, 0, 0], 5, axis = 1)
	surfs = fsc.flt.calculateFourierModes(field, points, 200, targetError = 1e-3)['surfaces']
	
	surfs + surfs
	surfs - surfs
	surfs * 2
	2 * surfs
	surfs / 2
	surfs[0]
	surfs[1:3]
	surfs[:,None]
	surfs.shape
	
	angles = [0, np.pi]
	surfs.evaluate(phi = angles, theta = angles)

def test_pc_plane(field):
	points = [1.1, 0, 0]
	plane = {"orientation" : {"normal" : [0,1,0]}}
	
	points = fsc.flt.poincareInPhiPlanes(points, field, [plane], 10)
	assert points.shape[-1] == 20
	print(points)
	
	
def test_axcur(field):
	fsc.flt.axisCurrent(field, 1000, startPoint = [1.05, 0, 0], targetError = 1e-3)

def test_directions(field):
	p = [1.05, 0, 0]
	fsc.flt.trace(p, field, distanceLimit = 0.01, direction = 'forward')
	fsc.flt.trace(p, field, distanceLimit = 0.01, direction = 'backward')
	fsc.flt.trace(p, field, distanceLimit = 0.01, direction = 'cw')
	fsc.flt.trace(p, field, distanceLimit = 0.01, direction = 'ccw')

def test_field_values(field, grid):
	fsc.flt.fieldValues(field, grid)
