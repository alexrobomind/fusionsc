import fusionsc as fsc
import numpy as np

from pytest import approx, fixture

from fusionsc.serialization import dump, load

import time

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
		np.asarray([{"ABC" : "DEF"}]),
		
		fsc.service.MagneticField.newMessage({"invert" : {"sum" : []}}),
		fsc.service.FLTStopReason.get(0),
		
		np.asarray([
			fsc.service.FLTStopReason.get(0),
			fsc.service.FLTStopReason.get(1),
			fsc.service.FLTStopReason.get(2)
		])
	]
	
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