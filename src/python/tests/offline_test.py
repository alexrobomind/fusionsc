import fusionsc as fsc
from fusionsc.devices import w7x

from fusionsc.magnetics import MagneticConfig, CoilFilament
from fusionsc.geometry import Geometry

import pytest

@pytest.fixture
def offlineData():
	# Create a fake offline data that encodes a W7-X coil, a component, and a config
	mapping = {
		w7x.coilsDBConfig(0) : MagneticConfig(),
		w7x.coilsDBCoil(0) : CoilFilament({'sum' : []}),
		w7x.component(0) : Geometry()
	}
	
	return fsc.resolve.createOfflineData(mapping)

def test_updateOfflineData(offlineData):
	fsc.resolve.updateOfflineData(offlineData, {w7x.component(1) : Geometry()})

def test_createOfflineData(tmp_path, offlineData):
	fsc.data.writeArchive(offlineData, str(tmp_path / "offline.fsc"))
	fsc.resolve.importOfflineData(str(tmp_path / "offline.fsc"))
	
	fsc.asnc.wait(w7x.component(0).merge())
	
	fsc.resolve.reset()
	
	with pytest.raises(Exception):
		fsc.asnc.wait(w7x.component(0).merge())

def test_createOfflineWarehouse(tmp_path, offlineData):
	whPath = "sqlite" + (tmp_path / "offline.sqlite").as_uri()[4:]
	print(whPath)
	
	fsc.resolve.updateWarehouse(whPath, offlineData)
	fsc.resolve.updateWarehouse(whPath, {w7x.component(1) : Geometry()})
	fsc.resolve.connectWarehouse(whPath)
	
	fsc.asnc.wait(w7x.component(0).merge())
	w7x.coilsDBCoil(0).biotSavart().evaluateXyz([1, 1, 1])
	
	print(w7x.coilsDBCoil(0).resolve())
	
	fsc.resolve.reset()
	
	with pytest.raises(Exception):
		fsc.asnc.wait(w7x.component(0).merge())
	