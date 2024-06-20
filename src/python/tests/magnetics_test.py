import fusionsc as fsc
from fusionsc.devices import w7x, jtext

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
def surfaces(field):
	points = np.linspace([1.1, 0, 0], [1.2, 0, 0], 5, axis = 1)
	surfs = fsc.flt.calculateFourierModes(field, points, 200, targetError = 1e-3)['surfaces']
	return surfs

def test_filaments():
	fil1 = fsc.magnetics.CoilFilament.fromArray([
		[0, 1, 1, 0],
		[0, 0, 1, 1],
		[0, 0, 0, 0]
	])
	fil2 = fil1 + fil1

def test_eval(field):
	x = [1, 0, 0]
	field.evaluateXyz(x)
	field.interpolateXyz(x)
	field.evaluatePhizr(x)

@pytest.mark.parametrize("quantity", ["field", "flux"])
@pytest.mark.parametrize("useFFT", [False, True])
def test_compradmodes(field, surfaces, quantity, useFFT):
	field.calculateRadialModes(surfaces, quantity = quantity, useFFT = useFFT)
	field.calculateRadialModes(surfaces, field, quantity = quantity, useFFT = useFFT)
	
def test_ops(field):
	field + field
	field * 2
	2 * field
	field - field
	field / 2

def test_computed(field):
	grid, data = field.getComputed()
	fsc.magnetics.MagneticConfig.fromComputed(grid, data)
	
def test_dipoles():
	fsc.magnetics.MagneticConfig.fromDipoles(
		[[1], [0], [0]],
		[[1], [0], [0]],
		[0.01]
	)

def test_ec():
	pertField = jtext.islandCoils([1, 1, 1, 1, 1, 1])
	fsc.magnetics.extractCoils(pertField)
