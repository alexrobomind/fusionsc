"""This module extends the native storage capabilities to dynamically wrapped python objects"""

import collections.abc
import sys
import numpy as np
import abc
import contextlib
import contextvars

from typing import Any, Optional
from .wrappers import asyncFunction

from . import data
from . import service
from . import wrappers
from . import capnp

_endianMap = {
	'>': 'big',
	'<': 'little',
	'=': sys.byteorder,
	'|': 'not applicable',
}

_maxInt = 2**64
_minInt = -2**63 + 1
_maxLen = 2**29 # Maximum size for inline data

_pickleEnabled = contextvars.ContextVar('fusionsc.serialize._pickleEnabled', default = False)

@contextlib.contextmanager
def allowPickle():
	_pickleEnabled.set(True)
	yield None
	_pickleEnabled.set(False)

def wrap(obj: Any):
	if isinstance(obj, capnp.Object):
		return obj
	
	return dump(obj)

@asyncFunction
def unwrap(obj):
	if isinstance(obj, service.DynamicObject.Reader):
		return load.asnc(obj)
	
	if isinstance(obj, service.DynamicObject.Builder):
		return load.asnc(obj)
	
	return obj

def dump(obj: Any):
	return _dump(obj, service.DynamicObject.newMessage(), set())

def _dump(obj: Any, builder: Optional[service.DynamicObject.Builder], memoSet: set):
	key = id(obj)
	builder.memoKey = key
	
	if key in memoSet:
		builder.memoized = None
	
	elif isinstance(obj, str):
		builder.text = obj
		
	elif isinstance(obj, bytes):
		if len(obj) < _maxLen:
			builder.data = obj
		else:
			builder.bigData = data.publish(obj)
	
	elif isinstance(obj, wrappers.RefWrapper):
		builder.initRef().target = obj.ref
		builder.ref.wrapped = True
	
	elif isinstance(obj, service.DataRef):
		builder.initRef().target = obj
	
	elif isinstance(obj, capnp.Struct):
		ds = builder.initDynamicStruct()
		ds.schema = obj.type_.toProto()
		ds.data = obj
	
	elif isinstance(obj, capnp.Enum):
		de = builder.initDynamicEnum()
		de.schema = obj.type.toProto()
		de.value = obj.raw
	
	elif isinstance(obj, int):
		if obj >= 0 and obj < _maxInt:
			builder.uint64 = obj
		elif obj < 0 and obj > _minInt:
			builder.int64 = obj
		else:
			# Number of bits needed is no. of bits neede for abs
			# value + 1 due to 2-complement signed representation.
			# Rounded up needs another 7 bits.
			numBytes = (obj.bit_length() + 8) // 8
			builder.pythonBigInt = obj.to_bytes(length = numBytes, byteorder='little', signed = True)
	
	elif isinstance(obj, float):
		builder.double = obj
	
	elif isinstance(obj, complex):
		c = builder.initComplex()
		c.real = obj.real
		c.imag = obj.imag
		
	elif isinstance(obj, np.ndarray) and obj.dtype.kind in "biufcSU":
		dtype = obj.dtype
		
		array = builder.initArray()
		odt = array.dType
		
		if dtype.kind == "b":
			odt.bool = True
		elif dtype.kind in "iufc":
			numeric = odt.initNumeric()
			
			if dtype.kind == "i":
				numeric.base.signedInt = None
			elif dtype.kind == "u":
				numeric.base.unsignedInt = None
			elif dtype.kind == "c":
				numeric.base.complex = None
			else:
				numeric.base.float = None
			
			numeric.numBytes = dtype.itemsize
			numeric.littleEndian = (
				True
				if _endianMap[dtype.byteorder] == 'little'
				else False
			)
			
		elif dtype.kind in "SU":
			special = odt.initSpecial()
			
			special.length = dtype.itemsize
			
			if dtype.kind == "S":
				special.byteArray = None
			elif dtype.kind == "U":
				special.unicodeString = None
				special.length /= 4
			#elif dtype.kind == "m":
			#	special.timedelta = None
			#else:
			#	special.datetime = None
			
			special.littleEndian = (
				True
				if _endianMap[dtype.byteorder] == 'little'
				else False
			)			
		
		array.shape = obj.shape
		
		byteData = obj.tobytes()
		if len(byteData) < _maxLen:
			array.data = byteData
		else:
			array.bigData = data.publish(obj.tobytes())
	
	elif isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
		flat = obj.flatten()
		
		first = flat[0]
		if isinstance(first, capnp.Enum):
			firstType = first.type
			
			sameType = True
			for el in flat:
				if not isinstance(el, capnp.Enum) or el.type != firstType:
					sameType = False
					break
			
			if sameType:
				# Qualified to be an enum array
				enumArray = builder.initEnumArray()
				enumArray.schema = firstType.toProto()
				enumArray.shape = obj.shape
				enumArray.data = [el.raw for el in flat]
				
				return builder
		
		array = builder.initDynamicObjectArray()
		array.shape = obj.shape
		
		out = array.initData(len(flat))
		
		for i, el in enumerate(flat):
			_dump(el, out[i], memoSet)
		
	elif isinstance(obj, collections.abc.Sequence):
		out = builder.initSequence(len(obj))
		
		for i, el in enumerate(obj):
			_dump(el, out[i], memoSet)
	
	elif isinstance(obj, collections.abc.Mapping):
		out = builder.initMapping(len(obj))
		
		for i, (k, v) in enumerate(obj.items()):
			_dump(k, out[i].key, memoSet)
			_dump(v, out[i].value, memoSet)
	
	else:
		assert _pickleEnabled.get(), """
			Writing arbitrary objects requires pickling to be enabled explicitly. This is neccessary
			because pickle is an arbitrary execution format that poses substantial security
			risks. Please consider requesting the serialization format to be extended to
			meet your needs. ONLY ALLOW UNPICKLING ON DATA YOU TRUST.
			
			You can	enable pickling by running inside a "with fusionsc.serialize.allowPickle()"
			statement. You will also have to do this for deserialization.
			"""
		
		import pickle
		builder.pythonPickle = pickle.dumps(obj)
	
	memoSet.add(key)

	return builder

@asyncFunction
async def _load(reader: service.DynamicObject.Reader, memoDict: dict):
	which = reader.which_()
	
	if which == "memoized":
		key = reader.memoKey
		
		assert key in memoDict, "Could not locate memoized object. This implies an incorrect store/load ordering"
		return memoDict[key]
	
	if reader.memoKey != 0:
		memoDict[reader.memoKey] 
		
	if which == "text":
		return str(reader.text)
	
	if which == "data":
		return memoryview(reader.data)
	
	if which == "bigData":
		return await data.download.asnc(reader.bigData)
	
	if which == "sequence":
		return [await _load.asnc(x, memoDict) for x in reader.sequence]
	
	if which == "mapping":
		return {
			await _load.asnc(e.key, memoDict) : await _load.asnc(e.value, memoDict)
			for e in reader.mapping
		}
	
	if which == "ref":
		if reader.ref.wrapped:
			return wrappers.RefWrapper(reader.ref.target)
		else:
			return reader.ref.target
	
	if which == "dynamicStruct":
		ds = reader.dynamicStruct
		return ds.data.interpretAs(capnp.Type.fromProto(ds.schema))
	
	if which == "uint64":
		return reader.uint64
	
	if which == "int64":
		return reader.int64
	
	if which == "double":
		return reader.double
	
	if which == "complex":
		return complex(reader.complex.real, reader.complex.imag)
	
	if which == "array":
		dtype = reader.array.dType
		
		whichType = dtype.which_()

		if whichType == "bool":
			typeStr = "b"
		elif whichType == "numeric":
			whichNumber = dtype.numeric.base.which_()
			
			if whichNumber == "float":
				baseNum = "f"
			elif whichNumber == "signedInt":
				baseNum = "i"
			elif whichNumber == "unsignedInt":
				baseNum = "u"
			elif whichNumber == "complex":
				baseNum = "c"
			else:
				raise ValueError("Unknown numeric type for numpy array")
			
			if dtype.numeric.littleEndian:
				endian = "<"
			else:
				endian = ">"
			
			typeStr = f"{endian}{baseNum}{dtype.numeric.numBytes}"
		elif whichType == "special":
			special = dtype.special
			whichSpecial = special.which_()
			
			if whichSpecial == "byteArray":
				formatCode = "S"
			elif whichSpecial == "unicodeString":
				formatCode = "U"
			#elif whichSpecial == "datetime":
			#	formatCode = "M"
			#elif whichSpecial == "timedelta":
			#	formatCode = "m"
			
			if dtype.special.littleEndian:
				endian = "<"
			else:
				endian = ">"
			
			typeStr = f"{endian}{formatCode}{dtype.special.length}"
		else:
			assert False, "Unknown DType for array"
		
		if reader.array.which_() == "data":
			data = reader.array.data
		else:
			data = await data.download.asnc(reader.array.bigData)
		
		return np.frombuffer(data, typeStr).reshape(reader.array.shape)
	
	if which == "enumArray":
		ea = reader.enumArray
		
		type = capnp.Type.fromProto(ea.schema)
		flat = [capnp.Enum.fromRaw(type, raw) for raw in ea.data]
		return np.asarray(flat).reshape(ea.shape)
	
	if which == "dynamicEnum":
		de = reader.dynamicEnum
		type = capnp.Type.fromProto(de.schema)
		return capnp.Enum.fromRaw(type, de.value)
	
	if which == "dynamicObjectArray":
		doa = reader.dynamicObjectArray
		flat = [await _load.asnc(e) for e in doa.data]
		return np.asarray(flat).reshape(doa.shape)
	
	if which == "pythonBigInt":
		return int.from_bytes(bytes = reader.pythonBigInt, byteorder="little", signed = True)
	
	if which == "pythonPickle":
		assert _pickleEnabled.get(), """
			For security reasons, unpickling objects must be explicitly enabled. ONLY
			DO THIS IF YOU 100% TRUST THE DATA YOU ARE READING. PICKLE IS AN EXECUTABLE
			FORMAT, NOT DATA.
		"""
		
		import pickle
		
		if reader.pythonPickle.which_() == "data":
			data = reader.pythonPickle.data
		else:
			data = await data.download.asnc(reader.pythonPickle.bigData)
			
		return pickle.loads(data)
			
	raise ValueError("I don't know how to interpret the given binary reader of type '" + which + "'")