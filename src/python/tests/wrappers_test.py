import fusionsc as fsc
from fusionsc.devices import w7x

import pytest

def test_toYaml():
	w7x.standard().toYaml()

def test_refWrappers(tmp_path):
	wrapper = fsc.wrappers.RefWrapper(fsc.data.publish(w7x.standard()))
	wrapper.download()
	
	tmp_file = str(tmp_path / "wrapperStorage.fsc")
	
	wrapper.save(tmp_file)
	
	fsc.wrappers.RefWrapper.load(tmp_file)
	
	import copy
	copy.copy(wrapper)
	copy.deepcopy(wrapper)
	
	w7x.standard().save(tmp_file)
	fsc.magnetics.MagneticConfig.load(tmp_file)