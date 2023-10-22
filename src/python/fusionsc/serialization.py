from service import DynamicObject
from . import data
import collections.abc
import sys
import wrappers

import numpy as np

_endianMap = {
	'>': 'big',
	'<': 'little',
	'=': sys.byteorder,
	'|': 'not applicable',
}

_maxInt = 2**64
_minInt = -2**63 + 1
_maxLen = 2**29 # Maximum size for inline data

def dump(obj, builder = None, pickle = False):
	if builder is None:
		builder = DynamicObject.newMessage()
		
	if isinstance(obj, str):
		builder.text = obj
		
	elif isinstance(obj, bytes):
		if len(obj) < _maxLen:
			builder.data = obj
		else:
			builder.bigData = data.publish(obj)
	
	elif isinstance(obj, wrappers.RefWrapper):
		builder.initRef().target = obj.ref
		builder.ref.wrapped = True
	
	elif isinstance(obj, capnp.DataRef):
		builder.initRef().target = obj
	
	elif isinstance(obj, capnp.StructReader) or isinstance(obj, capnp.StructBuilder):
		ds = builder.initDynamicStruct()
		ds.schema = obj.schema_
		ds.data = obj
	
	elif isinstance(obj, int):
		if obj >= 0 and obj < _maxInt:
			builder.uint64 = obj
		elif obj > _minInt:
			builder.int64 = obj
		else:
			builder.pythonBigInt = obj.to_bytes(byteorder='little', signed = True)
		
	elif isinstance(obj, np.array) and obj.dtype.kind in "biuf":
		dtype = obj.dtype
		
		array = builder.initArray()
		odt = array.initDType()
		
		if dtype.kind == "b":
			odt.bool = True
		elif dtype.kind in "iuf":
			numeric = odt.initNUmeric()
			
			if dtype.kind == "i":
				numeric.base.signedInteger = True
			elif dtype.kind == "u":
				numeric.base.unsignedInteger = True
			else:
				numeric.base.float = True
			
			numeric.littleEndian = (
				True
				if _endianMap[dtype.byteorder] == 'little'
				else False
			)
			
			numeric.numBytes = dtype.itemsize
		
		array.shape = obj.shape
		array.data = data.publish(array.tobytes())
	
	elif isinstance(obj, np.array) and obj.dtype.kind == "O":
		array = builder.initObjectArray()
		array.shape = obj.shape
		
		flat = array.flatten()
		out = array.initData(len(flat))
		
		for i, el in enumerate(array.flatten()):
			serializeInto(el, out[i])
		
	elif isinstance(obj, collections.abc.Sequence):
		out = builder.initSequence(len(obj))
		
		for i, el in enumerate(obj):
			serializeInto(el, out[i])
	
	elif isinstance(obj, collections.abc.Mapping):
		out = builder.initMapping(len(obj))
		
		for i, (k, v) in enumerate(obj.items()):
			serializeInto(k, out[i].key)
			serializeInto(v, out[i].value)
	
	else:
		assert pickle, """
			Writing arbitrary objects requires pickling to be enabled.
			Please keep in mind that this will require the target to also
			enable pickle-based deserialization (which is disabled by default)
			"""
		
		import pickle
		builder.pythonPickle = pickle.dumps(obj)

	return builder

@asyncFunction
async def load(reader, pickle = False):
	which = reader.which_()
	if which == "text":
		return reader.text
	
	if which == "data":
		return reader.data
	
	if which == "bigData":
		return await data.download(reader.bigData)
	
	if which == "sequence":
		return [await load(x, pickle) for x in reader.sequence]
	
	if which == "mapping":
		return {
			await load(e.key, pickle) : await load(e.value, pickle)
			for e in reader.mapping
		}
	
	if which == "ref":
		if reader.ref.wrapped:
			return wrappers.RefWrapper(reader.ref.target)
		else:
			return reader.ref.target
	
	if which == "dynamicStruct":
		ds = reader.dynamicStruct
		return ds.data.interpretAs(ds.schema)
	
	if which == "uint64":
		return reader.uint64
	
	if which == "int64":
		return reader.int64
	
	if which == "double":
		return reader.double
	
	if which == "array":
		dtype = reader.array.dType

		if dtype.isBool():
			typeStr = "b"
		elif dtype.isNumeric():
			whichNumber = dtype.numeric.which()
			
			if whichNumber == "float":
				baseNum = "f"
			elif whichNumber == "signedInteger":
				baseNum = "i"
			else:
				baseNum = "u"
			
			if dtype.numeric.littleEndian:
				endian = "<"
			else:
				endian = ">"
			
			typeStr = f"{endian}{baseNum}{dtype.numeric.numBytes}"
		else:
			assert False, "Unknown DType for array"
		
		if reader.array.which_() == "data":
			data = reader.array.data
		else:
			data = await data.download(reader.array.bigData)
		
		return np.frombuffer(data, typeStr).reshape(reader.array.shape)
	
	if which == "dynamicObjectArray":
		return np.asarray([await load(e, pickle) for e in reader.objectArray.data]).reshape(reader.objectArray.shape)
	
	if which == "pythonBigInt":
		return int.from_bytes(byteorder="little", signed = True)
	
	if which == "pythonPickle":
		assert pickle, "Loading from pickle requires explicit permission (due to the associated security risks)"
		import pickle
		
		if reader.pythonPickle.which_() == "data":
			data = reader.pythonPickle.data
		else:
			data = await data.download(reader.pythonPickle.bigData)
			
		return pickle.loads(data)
			
	raise ValueError("I don't know how to interpret the given binary reader of type '" + which + "'")