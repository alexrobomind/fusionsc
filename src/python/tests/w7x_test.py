import fusionsc as fsc
from fusionsc.devices import w7x

import pytest
import asyncio

# We can not compute the fields, but we can test the abstract specification
def test_configs():
	w7x.standard()
	w7x.highMirror()
	w7x.highIota()
	w7x.lowIota()
	
	w7x.coilsDBConfig(5)
	w7x.coilsDBCoil(5)

def test_cadCoils():
	grid = w7x.defaultGrid()
	grid.nR = 5
	grid.nPhi = 4
	grid.nZ = 3
	
	coils = w7x.cadCoils()
	coils = coils.computeFields(grid)
	
	field = w7x.standard(coils = coils) + w7x.trimCoils(coils = coils) + w7x.controlCoils(coils = coils)
	

def test_geo():
	grid = w7x.defaultGeometryGrid()
	grid.nX = 5
	grid.nY = 4
	grid.nZ = 3
	geo1 = w7x.op12Geometry().index(grid)
	geo2 = w7x.op21Geometry().index(grid)