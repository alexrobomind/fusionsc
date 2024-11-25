import fusionsc as fsc
import numpy as np

from pytest import approx, fixture

from fusionsc.serialize import dump, load

import time
import pytest
import pickle

@fsc.serialize.cls()
class PickleDummy:
	data: str
	msg: fsc.service.Float64Tensor.Builder

def test_serialize_pickle():
	testString = 'test string'
	dummy1 = PickleDummy()
	dummy1.data = testString
	dummy1.msg = fsc.service.Float64Tensor.newMessage([1, 2, 3])
	
	dummy2 = PickleDummy()
	dummy2.data = "ABC"
	dummy2.msg = dummy2
	
	assert not fsc.serialize.pickleEnabled()
	
	dumped = fsc.serialize.dump((dummy1, dummy2))
	print(dumped)
	print(dumped.totalBytes_())
	
	b = dumped.canonicalize_()
	
	import zlib
	print(len(b), len(zlib.compress(b)))
	print(dumped.canonicalize_())
	
	b2 = pickle.dumps((dummy1, dummy2))
	print(len(b2), len(zlib.compress(b2)))
	
	loaded, _ = fsc.serialize.load(dumped)
	
	assert loaded.data == testString
	

def test_serialize_complex():
	recursiveList = []
	recursiveList.append(recursiveList)
	
	recursiveDict = {}
	recursiveDict[0] = recursiveDict
	
	values = [
		np.asarray([fsc.service.MagneticField.newMessage(), fsc.service.MagneticField.newMessage({"invert" : None})], dtype = object),
		recursiveList,
		recursiveDict
	]
	
	for val in values:
		dumped = dump(val)
		read = load(dumped)
		
		print(dumped)
		print(type(val))
		print(val)
		print(type(read))
		print(read)

def test_serialize_simple():
	values = [
		0,
		1,
		-1,
		2**64,
		2**128,
		-2**30,
		-2**128,
		b'ABCDE',
		'EDCBA',
		1.0,
		complex(1, 2),
		np.asarray([3]),
		np.asarray([complex(3, 3)]),
		np.asarray([3, 5.0]),
		np.asarray(["Hi","there"]),
		np.asarray([b'ABC', b'DEF', b'G']),
		np.asarray([{"ABC" : "ABC"}]),
		
		np.ones(2, [('x', np.uint16, (2, 2)), ('y', np.float32)]),
		
		fsc.service.MagneticField.newMessage({"invert" : {"sum" : []}}),
		fsc.service.FLTStopReason.get(0),
		
		np.asarray([
			fsc.service.FLTStopReason.get(0),
			fsc.service.FLTStopReason.get(1),
			fsc.service.FLTStopReason.get(2),
			fsc.service.FLTStopReason.get(200)
		]),
		
		[4, 4, 4]
	]
	
	print('Initing')
	
	for val in values:
		dumped = dump(val)
		read = load(dumped)
		print(dumped)
		print(type(val))
		print(val)
		print(type(read))
		print(read)
		
		if isinstance(val, np.ndarray):
			assert isinstance(read, np.ndarray)
			assert read.dtype == val.dtype
			assert np.array_equal(val, read)
		elif isinstance(val, fsc.capnp.Struct):
			assert(type(val) == type(read))
			assert str(read) == str(val)
		else:
			assert read == val
