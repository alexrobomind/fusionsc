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
	coils = w7x.cadCoils()
	coils = coils.computeFields(w7x.defaultGrid())
	
	field = w7x.standard(coils = coils) + w7x.trimCoils(coils = coils) + w7x.controlCoils(coils = coils)
	
	# We have no resolve mechanisms for W7-X
	#with pytest.raises(Exception):
	#	asyncio.run(coils)

def test_geo():
	geo1 = w7x.op12Geometry().index(w7x.defaultGeometryGrid())
	geo2 = w7x.op21Geometry().index(w7x.defaultGeometryGrid())