import fusionsc as fsc

from pytest import approx, fixture

from fusionsc.serialization import dump, load

def test_serialize_int():
	values = [0, 1, -1, 2**64, 2**128, -2**30, -2**128]
	
	for val in values:
		dumped = dump(val)
		print(dumped)
		#read = load(val)
		read = load(dumped)
		
		assert read == val