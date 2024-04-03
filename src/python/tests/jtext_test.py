import fusionsc as fsc
from fusionsc.devices import jtext

import pytest

import asyncio

@pytest.fixture
def grid():
	g = jtext.defaultGrid()
	g.nR = 3
	g.nZ = 4
	g.nPhi = 5
	
	return g
	

def test_geo():
	asyncio.run(jtext.pfcs(0.24).index(jtext.defaultGeometryGrid()))

def test_coils(grid):
	asyncio.run(jtext.islandCoils([1] * 6).compute(grid))

def test_equi(grid):
	field = fsc.magnetics.MagneticConfig.fromEFit(jtext.exampleGeqdsk())
	asyncio.run(field.compute(grid))