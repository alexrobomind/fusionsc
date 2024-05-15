"""This module extends the native storage capabilities to dynamically wrapped python objects"""

import collections.abc
import sys
import numpy as np
import abc
import contextlib
import contextvars

from typing import Any, Optional
from .asnc import asyncFunction
from .wrappers import structWrapper

from . import data
from . import service
from . import wrappers
from . import capnp
from . import native

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
_pickleBlocked = False

import os
if 'FUSIONSC_BLOCK_PICKLE' in os.environ and os.environ['FUSIONSC_BLOCK_PICKLE'] != '0':
	_pickleBlocked = True

def blockPickle():
	_pickleBlocked = True

class UnknownObject(structWrapper(service.DynamicObject)):
	pass

@contextlib.contextmanager
def allowPickle():
	if _pickleBlocked:
		import warnings
		warning.warn("""
			An attempt to enable pickling while pickling was blocked because either fsc.serialize.blockPickle()
			was called or because the environment variable FUSIONSC_BLOCK_PICKLE was set to a value
			other than '0'.
		""")
	
	token = _pickleEnabled.set(True)
	yield None
	_pickleEnabled.reset(token)

def pickleEnabled():
	"""Returns whether pickling is allowed in the current context"""
	return _pickleEnabled.get() and not _pickleBlocked

def wrap(obj: Any):
	"""Converts python objects into a service.DynamicObject.Builder but returns native messages as-is"""
	if isinstance(obj, capnp.Object):
		return obj
	
	return dump(obj)

@asyncFunction
async def unwrap(obj):
	"""Unfolds service.DynamicObject instances into python objects and returns others as-is"""
	if isinstance(obj, service.DynamicObject.Reader):
		return await load.asnc(obj)
	
	if isinstance(obj, service.DynamicObject.Builder):
		return await load.asnc(obj)
	
	return obj

def dump(obj: Any):
	"""
	Converts python objects into nested structures of service.DynamicObject
	"""
	return _dump(obj, service.DynamicObject.newMessage(), set(), set())

def _dumpDType(dtype, odt: service.DynamicObject.DType.Builder):
	assert dtype.kind != "O", "Can not export objects in primitive arrays (e.g. as struct fields)"
	
	if dtype.subdtype is not None:
		itemType, shape = dtype.subdtype
		outArr = odt.initSubArray()
		
		_dumpDType(itemType, outArr.itemType)
		outArr.shape = shape
		
		return
	
	if dtype.fields is not None:
		outFields = odt.initStruct().initFields(len(dtype.fields))
		
		for i, (name, info) in enumerate(dtype.fields.items()):
			field = outFields[i]
			field.name = name
			field.offset = info[1]
			_dumpDType(info[0], field.dType)
		
		return
	
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
	else:
		struct = odt.init

def _loadDType(dtype):	
	whichType = dtype.which_()

	if whichType == "bool":
		return "b"
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
		
		return f"{endian}{baseNum}{dtype.numeric.numBytes}"
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
		
		return f"{endian}{formatCode}{dtype.special.length}"
	elif whichType == "subArray":
		sa = dtype.subArray
		return (_loadDType(sa.itemType), tuple(sa.shape))
	elif whichType == "struct":
		return {
			str(field.name) : (_loadDType(field.dType), field.offset) 
			for field in dtype.struct.fields
		}
	else:
		assert False, "Unknown DType for array"

def _dump(obj: Any, builder: Optional[service.DynamicObject.Builder], memoSet: set, recMemoSet: set):
	key = id(obj)
	builder.memoKey = key
	
	if key in memoSet:
		builder.memoized = None
		return builder
	
	if key in recMemoSet:
		builder.memoizedParent = None
		return builder
	
	recMemoSet.add(key)
	
	if isinstance(obj, UnknownObject):
		builder.nested = obj.data
	
	elif isinstance(obj, str):
		builder.text = obj
		
	elif isinstance(obj, bytes):
		if len(obj) < _maxLen:
			builder.data = obj
		else:
			builder.bigData = data.publish(obj)
	
	elif isinstance(obj, wrappers.RefWrapper):
		builder.initRef().target = obj.ref.castAs_(service.DataRef)
		builder.ref.wrapped = True
	
	elif isinstance(obj, service.DataRef.Client):
		builder.initRef().target = obj.ref.castAs_(service.DataRef)
	
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
		
	elif isinstance(obj, np.ndarray) and obj.dtype.kind in "biufcSUV":
		dtype = obj.dtype
		
		array = builder.initArray()
		array.shape = obj.shape
		
		_dumpDType(dtype, array.dType)		
		
		byteData = obj.tobytes()
		if len(byteData) < _maxLen:
			array.data = byteData
		else:
			array.bigData = data.publish(obj.tobytes())
	
	elif isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
		def handleObjectArray():
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
					
					return
			
			if isinstance(first, capnp.Struct):
				firstType = first.type_
				
				sameType = True
				for el in flat:
					if not isinstance(el, capnp.Struct) or el.type_ != firstType:
						sameType = False
						break
				
				if sameType:
					# Qualified to be a homogenous struct array
					structArray = builder.initStructArray()
					structArray.schema = firstType.toProto()
					structArray.shape = obj.shape
					
					sData = structArray.initData(len(flat))
					for i in range(len(flat)):
						sData[i].target = flat[i]
					
					return
			
			array = builder.initDynamicObjectArray()
			array.shape = obj.shape
			
			out = array.initData(len(flat))
			
			for i, el in enumerate(flat):
				_dump(el, out[i], memoSet, recMemoSet)
		
		handleObjectArray()
		
	elif isinstance(obj, collections.abc.Sequence):
		out = builder.initSequence(len(obj))
		
		for i, el in enumerate(obj):
			_dump(el, out[i], memoSet, recMemoSet)
	
	elif isinstance(obj, collections.abc.Mapping):
		out = builder.initMapping(len(obj))
		
		for i, (k, v) in enumerate(obj.items()):
			_dump(k, out[i].key, memoSet, recMemoSet)
			_dump(v, out[i].value, memoSet, recMemoSet)
	
	elif hasattr(obj, "_fusionsc_wraps"):
		_dump(obj._fusionsc_wraps, builder, memoSet, recMemoSet)
	
	else:
		assert pickleEnabled(), """
			Writing arbitrary objects requires pickling to be enabled explicitly. This is neccessary
			because pickle is an arbitrary execution format that poses substantial security
			risks. Please consider requesting the serialization format to be extended to
			meet your needs. ONLY ALLOW UNPICKLING ON DATA YOU TRUST.
			
			You can	enable pickling by running inside a "with fusionsc.serialize.allowPickle()"
			statement. You will also have to do this for deserialization.
			"""
		
		import pickle
		data = pickle.dumps(obj)
		
		pickled = builder.initPythonPickle()
		if len(data) < _maxLen:
			pickled.data = data
		else:
			pickled.bigData = data.publish(pickled)
	
	memoSet.add(key)

	return builder

@asyncFunction
async def load(reader: service.DynamicObject.Reader):
	"""
	Loads a service.DynamicObject as a python object
	"""
	return await _load(reader, dict())

async def _load(reader: service.DynamicObject.Reader, memoDict: dict):
	result = await _interpret(reader, memoDict)
	
	if reader.memoKey != 0 and reader.memoKey not in memoDict:
		memoDict[reader.memoKey] = result
	
	return result

async def _interpret(reader: service.DynamicObject.Reader, memoDict: dict):
	which = reader.which_()
	
	if which == "memoized":
		key = reader.memoKey
		
		assert key in memoDict, "Could not locate memoized object. This implies an incorrect store/load ordering"
		return memoDict[key]
	
	if which == "memoizedParent":
		key = reader.memoKey
		
		assert key in memoDict, "Could not locate memoized parent object. This implies an incorrect store/load ordering"
		return memoDict[key]
		
	if which == "text":
		return str(reader.text)
	
	if which == "data":
		return memoryview(reader.data)
	
	if which == "bigData":
		return await data.download.asnc(reader.bigData)
	
	if which == "sequence":
		result = [None] * len(reader.sequence)
		
		if reader.memoKey != 0:
			assert reader.memoKey not in memoDict
			memoDict[reader.memoKey] = result
		
		for i, x in enumerate(reader.sequence):
			result[i] = await _load(x, memoDict)
		
		return result
	
	if which == "mapping":
		result = {}
		
		if reader.memoKey != 0:
			assert reader.memoKey not in memoDict
			memoDict[reader.memoKey] = result
		
		for e in reader.mapping:
			key = await _load(e.key, memoDict)
			val = await _load(e.value, memoDict)
			
			result[key] = val
		
		return result
	
	if which == "ref":
		if reader.ref.wrapped:
			return wrappers.RefWrapper(reader.ref.target)
		else:
			return reader.ref.target
	
	if which == "dynamicStruct":
		ds = reader.dynamicStruct
		print("Schema: ", ds.schema)
		print("Schema type: ", type(ds.schema))
		return ds.data.castAs(capnp.Type.fromProto(ds.schema))
	
	if which == "uint64":
		return reader.uint64
	
	if which == "int64":
		return reader.int64
	
	if which == "double":
		return reader.double
	
	if which == "complex":
		return complex(reader.complex.real, reader.complex.imag)
	
	if which == "array":
		typeDesc = _loadDType(reader.array.dType)
		
		if reader.array.which_() == "data":
			data = reader.array.data
		else:
			data = await data.download.asnc(reader.array.bigData)
		
		return np.frombuffer(data, typeDesc).reshape(reader.array.shape)
	
	if which == "enumArray":
		# Note: The following code does the job just fine,
		# but it turns out to be quite slow on large
		# arrays
		"""
		ea = reader.enumArray
		tp = capnp.Type.fromProto(ea.schema)
		flat = [capnp.Enum.fromRaw(tp, raw) for raw in ea.data]
		return np.asarray(flat).reshape(ea.shape)
		"""
		def lazyLoad():
			return native.serialize.loadEnumArray(reader.enumArray)
		return wrappers.LazyObject(lazyLoad)
	
	if which == "dynamicEnum":
		de = reader.dynamicEnum
		tp = capnp.Type.fromProto(de.schema)
		return capnp.Enum.fromRaw(tp, de.value)
	
	if which == "dynamicObjectArray":
		doa = reader.dynamicObjectArray
		result = np.empty([len(doa.data)], "O")
		for i in range(len(doa.data)):
			result[i] = await _load(doa.data[i], memoDict)
		
		return result.reshape(doa.shape)
	
	if which == "structArray":
		# Note: The following code does the job just fine,
		# but it turns out to be quite slow on large
		# arrays
		"""
		sa = reader.structArray
		tp = capnp.Type.fromProto(sa.schema)
		
		result = np.empty([len(sa.data)], "O")
		for i in range(len(sa.data)):
			result[i] = sa.data[i].target.castAs(tp)
		
		return result.reshape(sa.shape)
		"""
		def lazyLoad():
			return native.serialize.loadStructArray(reader.structArray)
		return wrappers.LazyObject(lazyLoad)
	
	if which == "pythonBigInt":
		return int.from_bytes(bytes = reader.pythonBigInt, byteorder="little", signed = True)
	
	if which == "pythonPickle":
		assert pickleEnabled(), """
			For security reasons, unpickling objects must be explicitly enabled. ONLY
			DO THIS IF YOU 100% TRUST THE DATA YOU ARE READING. PICKLE IS AN EXECUTABLE
			FORMAT, NOT PURE DATA.
		"""
		
		import pickle
		
		if reader.pythonPickle.which_() == "data":
			data = reader.pythonPickle.data
		else:
			data = await data.download.asnc(reader.pythonPickle.bigData)
			
		return pickle.loads(data)
	
	if which == "nested":
		return await _interpret(reader.nested, memoDict)
			
	import warnings
	warnings.warn(
		"""An unknown type of dynamic object was encountered. This usually means that
		the data being loaded were saved by a newer version of this library. The
		'UnknownObject' returned contains all information stored in this section of
		the message and can still be saved (the data will be written back as encountered
		here"""
	)
	
	return UnknownObject(reader)