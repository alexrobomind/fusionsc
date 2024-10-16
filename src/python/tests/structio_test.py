import fusionsc as fsc
from fusionsc.devices import w7x, jtext

import pytest
import numpy as np

@pytest.fixture
def siodata():
	coil = fsc.magnetics.CoilFilament.fromArray([
		[0, 1], [1, 2], [2, 3]
	])
	
	return {
		"cadCoils" : w7x.cadCoils(),
		"seq" : [1, "Hi"],
		"coil" : coil
	}

@pytest.mark.parametrize("lang", ['json', 'yaml', 'cbor', 'bson', 'ubjson', 'msgpack'])
def test_dumps(siodata, lang):
	structure = {
		"cadCoils" : w7x.CoilPack({"fields" : None})
	}
	
	print("Original:", siodata)
	dumped = fsc.structio.dumps(siodata, lang)
	print("Dump:", dumped)
	loaded = fsc.structio.load(dumped, lang = lang)
	print("Loaded:", loaded)
	loaded = fsc.structio.load(dumped, structure, lang = lang)
	print("Loaded (w/ unification):", loaded)

@pytest.mark.parametrize("lang", ['json', 'yaml', 'cbor', 'bson', 'ubjson', 'msgpack'])
def test_recdumps(siodata, lang):
	structure = {
		"cadCoils" : w7x.CoilPack({"fields" : None}),
		"coil" : fsc.magnetics.CoilFilament()
	}
	
	print("Original:", siodata)
	dumped = fsc.structio.recursiveDumps(siodata, lang)
	print("Dump:", dumped)
	loaded = fsc.structio.load(dumped, lang = lang)
	print("Loaded:", loaded)
	loaded = fsc.structio.load(dumped, structure, lang = lang)
	print("Loaded (w/ unification):", loaded)

@pytest.mark.parametrize("lang", ['json', 'yaml', 'cbor', 'bson', 'ubjson', 'msgpack'])
def test_dump(siodata, lang, tmp_path):
	filename = str(tmp_path / 'structio_test')
	with open(filename, 'wb') as f:
		fsc.structio.dump(siodata, f, lang)
	
	with open(filename, 'rb') as f:
		loaded = fsc.structio.load(f, None, lang)
		
	dumped = fsc.structio.dumps(siodata, lang)
	loaded = fsc.structio.load(dumped, lang = lang)
	print("Loaded:", loaded)

@pytest.mark.parametrize("lang", ['json', 'yaml', 'cbor', 'bson', 'ubjson', 'msgpack'])
def test_recdump(siodata, lang, tmp_path):
	filename = str(tmp_path / 'structio_test')
	with open(filename, 'wb') as f:
		fsc.structio.recursiveDump(siodata, f, lang)
	
	with open(filename, 'rb') as f:
		loaded = fsc.structio.load(f, None, lang)
		
	dumped = fsc.structio.dumps(siodata, lang)
	loaded = fsc.structio.load(dumped, lang = lang)

@pytest.mark.parametrize("lang", ['json', 'yaml', 'cbor', 'bson', 'ubjson', 'msgpack'])
def test_nparray1(lang):
	data = np.asarray([["Hi", 2], [3, 4]])
	
	dumped = fsc.structio.dumps(data, lang)
	print("Dump:", dumped)
	
	dst = fsc.structio.load(dumped, lang = lang)
	print(data, data.shape, data.dtype)

@pytest.mark.parametrize("lang", ['json', 'yaml', 'cbor', 'bson', 'ubjson', 'msgpack'])
def test_nparray2(lang):
	data = np.asarray([1, 2, 3, 4])
	
	dumped = fsc.structio.dumps(data, lang)
	print("Dump:", dumped)
	
	dst = fsc.structio.load(dumped, lang = lang)
	print(data, data.shape, data.dtype)

@pytest.mark.parametrize("lang", ['json', 'yaml', 'cbor', 'bson', 'ubjson', 'msgpack'])
def test_npfake1(lang):
	data = {
		"shape" : [1, 2, "Hi"],
		"data" : [1, 2, 3, 4]
	}
	
	dumped = fsc.structio.dumps(data, lang)
	print("Dump:", dumped)
	
	dst = fsc.structio.load(dumped, lang = lang)
	print(data)

@pytest.mark.parametrize("lang", ['json', 'yaml', 'cbor', 'bson', 'ubjson', 'msgpack'])
def test_npfake2(lang):
	data = {
		"shape" : [2, 2],
		"data" : [1, 2, 3, 4, 5]
	}
	
	dumped = fsc.structio.dumps(data, lang)
	print("Dump:", dumped)
	
	dst = fsc.structio.load(dumped, lang = lang)
	print(data)