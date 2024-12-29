"""
This module extends the native storage capabilities to dynamically wrapped python objects

It operates by converting python objects to and from a special service type (service.DynamicObject).
Common scientific types (mappings, sequences, numpy arrays, various number types) are supported
natively with an emphasis on fast storage and retrieval. More complex types are serialized using
the __reduce__ / __reduce_ex__ protocol that is also employed by pickle.

Unlike pickle, this module does not allow arbitrary calls during deserialization (unless the unsafe
enablePickle() mode is used to match pickle's behavior 1:1). Classes and deserialization helpers
have to be imported and registered with this module before attempting deserialization.
"""

import collections.abc
import sys
import numpy as np
import abc
import contextlib
import contextvars
import pickle
import weakref

from typing import Any, Optional, Literal
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

_loadMode = contextvars.ContextVar("fusionsc.serialize._loadMode", default = "fast")

_globalsDict = {}
_globalInfo = {}

def register(x = None, mod = None, name = None, mayCall = False):
	def f(x):
		global _globalsDict, _globalInfo
		
		modActual = mod if mod else x.__module__
		nameActual = name if name else x.__name__
		
		key = (modActual, nameActual)
		_globalsDict[key] = x
		_globalInfo[id(x)] = x, modActual, nameActual, mayCall
		
		return x
	
	return f(x) if x is not None else f

def cls(x = None, mod = None, name = None):
	"""Register a class that may be used for serialization and deserialization"""
	return register(x, mod, name, True)

def func(x = None, mod = None, name = None):
	"""Register a function to be used during deserialization (e.g. as a state setter)"""
	return register(mod, name, True)

def val(x = None, mod  = None, name = None):
	"""Register a non-callable data object usable during deserialization"""
	return register(mod, name, False)
		

import os
if 'FUSIONSC_BLOCK_PICKLE' in os.environ and os.environ['FUSIONSC_BLOCK_PICKLE'] != '0':
	_pickleBlocked = True

def blockPickle():
	"""Globally disables the usage of pickle for deserialization programatically"""
	_pickleBlocked = True

def _checkCallable(o):
	if(pickleEnabled()):
		return
	assert id(o) in _globalInfo, "Serialization / deserialization attempted to call an unregistered object. Try enabling pickling to make this work nonetheless."
	
	obj, mod, name, mayCall = _globalInfo[id(o)]
	
	assert mayCall, "Deserialization attempted to call an object not registered as callable. Try enabling pickling to make this work nonetheless."		

class UnknownObject(structWrapper(service.DynamicObject)):
	"""
	Wrapper for objects that could not be interpreted (usually because they were written
	by a future version of this module)
	"""
	pass

@contextlib.contextmanager
def allowPickle():
	"""
	Temporarily enables pickling and unpickling in the serialization engine.
	
	Serialization and deserialization are available as a fallback path for the serialization engine
	in case it encounters objects it can not process. However, this path is disabled by default
	because pickle allows arbitrary code to be called during deserialization. This is
	unexpected for data files, and can pose serious security issues if these files are loaded
	from untrusted sources.
	
	If blockPickle() was called or the environment variable FUSIONSC_BLOCK_PICKLE is set
	to another value than 0, this function will fail to enable pickling.
	"""
	if _pickleBlocked:
		import warnings
		warning.warn("""
			An attempt to enable pickling while pickling was without effect because either fsc.serialize.blockPickle()
			was called or because the environment variable FUSIONSC_BLOCK_PICKLE was set to a value
			other than '0'.
		""")
	
	token = _pickleEnabled.set(True)
	yield None
	_pickleEnabled.reset(token)

@contextlib.contextmanager
def _setLoadMode(mode):
	"""
	A context manager to change the loading mode (see doc. of "load")
	in nested function calls for strict processing inside objects serialized
	with the pickle-style protocol.
	"""
	token = _loadMode.set(mode)
	yield None
	_loadMode.reset(token) 

def pickleEnabled():
	"""Returns whether pickling is allowed in the current context"""
	return _pickleEnabled.get() and not _pickleBlocked

def wrap(obj: Any) -> capnp.Object:
	"""Converts python objects into a service.DynamicObject.Builder but returns native messages as-is"""
	if isinstance(obj, capnp.Object):
		return obj
	
	return dump(obj)

@asyncFunction
async def unwrap(obj: capnp.Object, mode: Literal["strict", "fast"] = "fast") -> Any:
	"""Unfolds service.DynamicObject instances into python objects and returns others as-is"""
	if isinstance(obj, service.DynamicObject.Reader):
		return await load.asnc(obj, mode)
	
	if isinstance(obj, service.DynamicObject.Builder):
		return await load.asnc(obj, mode)
	
	return obj
		

def dump(obj: Any) -> service.DynamicObject.Builder:
	"""
	Converts python objects into nested structures of service.DynamicObject
	"""
	return _dump(obj, service.DynamicObject.newMessage(), _MemoSet(), _MemoSet())

@asyncFunction
async def load(reader: service.DynamicObject.ReaderOrBuilder, mode: Literal["strict", "fast"] = "fast") -> Any:
	"""
	Loads a service.DynamicObject as a python object.
	
	@warning When possible, this method will process the underlying object as a VIEW, not as a copy.
	If the data are loaded from a builder, the returned buffers will be mutable (numpy arrays will be set
	to read-only but that flag can be flipped as well by the user). Changes to these buffers will possibly
	be reflected in the builder and all views obtained through "load". Care should be taken in particular
	when modifying the buffers inside memoryview objects. bytes() and bytearray() objects as well as numpy
	arrays returned as mutable own their respective memory buffers and will not be affected.
	
	params:
	  mode: A mode that describes how to restore saved objects. The two options are:
	    - fast (default): In this mode, objects will be deserialized mostly as they are put in,
	      with the notable exception of buffer types, which will be restored as non-owned read-only
	      buffers wherever this saves buffer copies. In particular, the following transformations are applied:
	        - bytes, bytearray, memoryview, and PickleBuffer -> memoryview (read-only)
	        - numpy array of primitives -> read-only numpy array
	      
	      When loading objects saved through the pickle protocol, the loader will implicitly switch to
	      "strict" loading mode for these objects.
	      
	    - strict: In this mode, objects are restored with identical types. This can slow
	      down the loading process, as a lot of buffer-types in python (bytes, bytearray,
	      mutable numpy arrays) require copies of the underlying buffers to be made.
	"""
	with _setLoadMode(mode):
		return await _load(reader, dict())

class _MemoSet:
	"""
	Memoization mapping for python objects
	"""
	def __init__(self):
		self.data = {}
	
	def __contains__(self, o):
		key = id(o)
		
		if key not in self.data:
			return False
		
		#isStrong, ref = self.data[key]
		#
		#if isStrong:
		#	return ref is o
		#else:
		#	return ref() is o
		return True
		
	def add(self, x):
		# Note: The shenanigans with weakrefs can cause non-unique IDs to be generated.
		# While it is possible to handle this, I decided that working this into the code
		# creates maintenance burden outweighing the speedups.
		
		#try:
		#	self.data[id(x)] = (False, weakref.ref(x))
		#except TypeError:
		#	self.data[id(x)] = (True, x)
		
		self.data[id(x)] = x
	
def _dumpReduced(obj: Any, po: service.DynamicObject.PythonObject.Builder, memoSet: _MemoSet, recMemoSet: _MemoSet, func, args, state = None, listItems = None, dictItems = None, stateSetter = None):
	"""
	Helper method to dump a result of __reduce__ or __reduce_ex__
	"""
	
	newobj = None
	
	if func.__name__ == "__newobj__":
		cls = args[0]
		newArgs = args[1:]
		kwargs = {}
		newobj = (cls, newArgs, {})
	elif func.__name == "__newobj_ex__":
		newobj = args
	
	if newobj is not None:
		cls, newArgs, newKwargs = newobj
		
		no = po.createBy.initNewobj()
		_dump(cls, no.cls, memoSet, recMemoSet)
		
		if newArgs:
			argsOut = no.initArgs(len(newArgs))
			for i, v in enumerate(newArgs):
				_dump(v, argsOut[i], memoSet, recMemoSet)
		
		if newKwargs:
			kwargsOut = no.initKwargs(len(newKwargs))
			for i, (k, v) in enumerate(newKwargs.items()):
				_dump(k, kwargsOut[i].key, memoSet, recMemoSet)
				_dump(k, kwargsOut[i].value, memoSet, recMemoSet)
	else:
		c = po.createBy.initCall()
		_dump(func, c.func, memoSet, recMemoSet)
		
		argsOut = c.initArgs(len(args))
		for i, v in enumerate(args):
			_dump(v, argsOut[i], memoSet, recMemoSet)
	
	# Object is created here during deserialization, so we can add the rec key back in
	recMemoSet.add(obj)
	
	if listItems:
		listItemsOut = po.initListItems(len(listItems))
		for i, v in enumerate(listItems):
			_dump(v, listItemsOut[i], memoSet, recMemoSet)
	
	if dictItems:
		dictItemsOut = po.initDictItems(len(dictItems))
		for i, (k, v) in enumerate(dictItems.items()):
			o = dictItemsOut[i]
			_dump(k, o.key, memoSet, recMemoSet)
			_dump(k, o.value, memoSet, recMemoSet)
	
	if state is not None:
		if stateSetter is None:
			_dump(state, po.state.initSetState(), memoSet, recMemoSet)
		else:
			ws = po.state.initWithSetter()
			_dump(stateSetter, ws.setter, memoSet, recMemoSet)
			_dump(state, ws.value, memoSet, recMemoSet)

def _dumpDType(dtype: np.dtype, odt: service.DynamicObject.DType.Builder):
	"""Helper method to dump a numpy dtype object"""
	
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

def _dump(obj: Any, builder: service.DynamicObject.Builder, memoSet: _MemoSet, recMemoSet: _MemoSet):
	key = id(obj)
	builder.memoKey = key
	
	if obj in memoSet:
		builder.memoized = None
		return builder
	
	if obj in recMemoSet:
		builder.memoizedParent = None
		return builder
	
	def saveBuffer(obj):
		if len(obj) < _maxLen:
			return { "data" : obj }
		else:
			return { "bigData" : data.publish(obj) }
	
	if obj is None:
		builder.pythonNone = None
	elif obj is NotImplemented:
		builder.pythonNotImplemented = None
	elif obj is Ellipsis:
		builder.pythonEllipsis = None
		
	# Registered globals are stored by name
	elif id(obj) in _globalInfo:
		_, mod, name, _ = _globalInfo[id(obj)]
		g = builder.initPythonGlobal()
		g.mod = mod
		g.name = name
	
	elif isinstance(obj, UnknownObject):
		builder.unknownObject = obj.data
	
	elif isinstance(obj, str):
		builder.text = obj
		
	elif isinstance(obj, bytes):
		builder.pythonWrapper = {
			"wrapped" : saveBuffer(obj),
			"bytes" : None
		}
	elif isinstance(obj, bytearray):
		builder.pythonWrapper = {
			"wrapped" : saveBuffer(obj),
			"bytearray" : None
		}
	elif isinstance(obj, memoryview):
		if obj.readonly:
			builder.nested = saveBuffer(obj)
		else:
			builder.pythonWrapper = {
				"wrapped" : saveBuffer(obj),
				"bytes" : None
			}
	elif isinstance(obj, pickle.PickleBuffer):
		builder.pythonWrapper = "pickleBuffer"
		_dump(memoryview(obj), builder.pythonWrapper.wrapped, memoSet, recMemoSet)
	
	#elif isinstance(obj, wrappers.RefWrapper):
	#	builder.initRef().target = obj.ref.castAs_(service.DataRef)
	#	builder.ref.wrapped = True
	
	elif isinstance(obj, service.DataRef.Client):
		builder.initRef().target = obj.ref.castAs_(service.DataRef)
	
	elif isinstance(obj, capnp.CapabilityClient):
		dc = builder.initDynamicCapability()
		dc.schema = obj.type_.toProto()
		dc.raw = obj
	
	elif isinstance(obj, capnp.Struct):
		ds = builder.initDynamicStruct()
		ds.schema = obj.type_.toProto()
		ds.data = obj
	
	elif isinstance(obj, capnp.Enum):
		de = builder.initDynamicEnum()
		de.schema = obj.type.toProto()
		de.value = obj.raw
	
	elif isinstance(obj, bool):
		builder.bool = obj
	
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
		
		mutable = obj.flags.writeable
		
		if mutable:
			wrapper = builder.initPythonWrapper()
			wrapper.mutableArray = None
			array = wrapper.wrapped.initArray()
		else:
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
			
			# Generic dump can contain recursive ref
			recMemoSet.add(obj)
			
			array = builder.initDynamicObjectArray()
			array.shape = obj.shape
			
			out = array.initData(len(flat))
			
			for i, el in enumerate(flat):
				_dump(el, out[i], memoSet, recMemoSet)
		
		handleObjectArray()
		
	elif isinstance(obj, (list, set, tuple)):
		parent = builder.initSequence()
		
		# Record types
		if isinstance(obj, tuple):
			parent.kind = "tuple"
		elif isinstance(obj, set):
			parent.kind = "set"
		
		if not isinstance(obj, tuple):
			recMemoSet.add(obj)
			
		out = parent.initContents(len(obj))
		for i, el in enumerate(obj):
			_dump(el, out[i], memoSet, recMemoSet)
	
	elif isinstance(obj, (dict,)):
		recMemoSet.add(obj)
		
		out = builder.initMapping(len(obj))
		
		for i, (k, v) in enumerate(obj.items()):
			_dump(k, out[i].key, memoSet, recMemoSet)
			_dump(v, out[i].value, memoSet, recMemoSet)
	
	#elif hasattr(obj, "_fusionsc_wraps"):
	#	_dump(obj._fusionsc_wraps, builder, memoSet, recMemoSet)
	
	elif hasattr(obj, "__module__") and hasattr(obj, "__qualname__") and pickleEnabled():
		g = builder.initPythonGlobal()
		g.mod = obj.__module__
		g.name = obj.__qualname__
	
	elif isinstance(obj, type) and not pickleEnabled():
		assert False, f"Types can not be serialized by default for safety reasons. Please register class {obj.__module__}.{obj.__name__} with the serialization engine or enable pickling for unsafe paths"
		
	elif hasattr(obj, "__reduce_ex__") or hasattr(obj, "__reduce__"):
		# Note: Object is added to recMemoSet inside _dumpReduced
		# at the appropriate moment.
		
		if hasattr(obj, "__reduce_ex__"):
			reduced = obj.__reduce_ex__(5)
		else:
			reduced = obj.__reduce__()
		_dumpReduced(obj, builder.initPythonObject(), memoSet, recMemoSet, *reduced)
	elif hasattr(obj, "__module__") and hasattr(obj, "__name__"):
		name = obj.__name__
		mod  = obj.__module__
		
		assert pickleEnabled(), f"""
			Serialization of this object requires the storage of the global object {mod}.{name}.
			This object has not been registered with FusionSC's serialization engine, therefore
			the object you are writing can only be read and written when full pickle compatibility
			is enabled.
			
			ENABLING THAT MODE IS A SECURITY RISK AS FILES CAN REQUEST ARBITRARY CODE DO BE EXECUTED.
			
			Preferably, please register {mod}.{name} with the serialization engine using the functions
			"cls", "var", or "func" in this module. This will whitelist them for deserialization as well.
			
			If this is not possible, you can enable full pickle compatibility by running inside a
			"with fusionsc.serialize.allowPickle():" statement.
			
			Please keep in mind that no matter which path you take, the person reading this object needs
			to take a similar path as well"""
			
		g = builder.initPythonGlobal()
		g.mod = mod
		g.name = name
		
	else:
		tp = type(obj)
		raise ValueError(f"No suitable method was found to serialize object of class {tp}")
		
#	else:
#		assert pickleEnabled(), """
#			Writing arbitrary objects requires pickling to be enabled explicitly. This is neccessary
#			because pickle is an arbitrary execution format that poses substantial security
#			risks. Please consider requesting the serialization format to be extended to
#			meet your needs. ONLY ALLOW UNPICKLING ON DATA YOU TRUST.
#			
#			You can	enable pickling by running inside a "with fusionsc.serialize.allowPickle()"
#			statement. You will also have to do this for deserialization.
#			"""
#		
#		import pickle
#		data = pickle.dumps(obj)
#		
#		pickled = builder.initPythonPickle()
#		if len(data) < _maxLen:
#			pickled.data = data
#		else:
#			pickled.bigData = data.publish(pickled)
	
	memoSet.add(obj)

	return builder

async def _load(reader: service.DynamicObject.ReaderOrBuilder, memoDict: dict):
	result = await _interpret(reader, memoDict)
	
	if reader.memoKey != 0 and reader.memoKey not in memoDict:
		memoDict[reader.memoKey] = result
	
	return result

def _loadDType(dtype):
	"""
	Helper method to load a serialized DType object as an array dtype descriptor
	"""
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

async def _interpret(reader: service.DynamicObject.ReaderOrBuilder, memoDict: dict):
	which = reader.which_()
		
	if which == "memoized"  or which == "memoizedParent":
		key = reader.memoKey
		
		if key not in memoDict and "unknown" in memoDict:
			import warnings
			warnings.warn("The deserialized object contains references that could not be resolved. " +
			"This is likely caused by a previous failure to deserialize the object properly, invalidating " +
			"the reference keys")
			
			return UnknownObject(reader)
		
		assert key in memoDict, "Could not locate memoized object. This implies an incorrect serialization or load ordering"
		return memoDict[key]
		
	if which == "text":
		return str(reader.text)
	
	if which == "data":
		return memoryview(reader.data)
	
	if which == "bigData":
		return memoryview(await data.download.asnc(reader.bigData))
	
	if which == "sequence":
		seqInfo = reader.sequence
		contents = seqInfo.contents
		
		if seqInfo.kind.which_() == "list":
			result = [None] * len(contents)
		
			if reader.memoKey != 0:
				assert reader.memoKey not in memoDict
				memoDict[reader.memoKey] = result
			
			for i, x in enumerate(contents):
				result[i] = await _load(x, memoDict)
		elif seqInfo.kind.which_() == "set":
			result = set()
		
			if reader.memoKey != 0:
				assert reader.memoKey not in memoDict
				memoDict[reader.memoKey] = result
				
			for x in contents:
				result.add(await _load(x, memoDict))
		elif seqInfo.kind.which_() == "tuple":
			result = tuple([
				await _load(x, memoDict)
				for x in contents
			])
		
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
		return ds.data.castAs(capnp.Type.fromProto(ds.schema))
	
	if which == "dynamicCapability":
		dc = reader.dynamicCapability
		return dc.raw.castAs_(capnp.Type.fromProto(dc.schema))
	
	if which == "uint64":
		return reader.uint64
	
	if which == "int64":
		return reader.int64
	
	if which == "double":
		return reader.double
	
	if which == "bool":
		return reader.bool
	
	if which == "complex":
		return complex(reader.complex.real, reader.complex.imag)
	
	if which == "array":
		typeDesc = _loadDType(reader.array.dType)
		
		if reader.array.which_() == "data":
			data = reader.array.data
		else:
			data = await data.download.asnc(reader.array.bigData)
		
		arr = np.frombuffer(data, typeDesc).reshape(reader.array.shape)
		
		# The arrays share the contents with the underlying buffer
		# This can create some oddities if it gets written to.
		arr.flags.writeable = False
		
		return arr
	
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
		return native.serialize.loadEnumArray(reader.enumArray)
	
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
		if _loadMode.get() == "fast":
			def lazyLoad():
				return native.serialize.loadStructArray(reader.structArray)
			return wrappers.LazyObject(lazyLoad)
		else:
			return native.serialize.loadStructArray(reader.structArray)
	
	if which == "pythonBigInt":
		return int.from_bytes(bytes = reader.pythonBigInt, byteorder="little", signed = True)
	
	if which == "pythonPickle":
		assert pickleEnabled(), """
			For security reasons, unpickling objects must be explicitly enabled. ONLY
			DO THIS IF YOU 100% TRUST THE DATA YOU ARE READING. PICKLE IS AN EXECUTABLE
			FORMAT, NOT PURE DATA.
		"""
				
		if reader.pythonPickle.which_() == "data":
			data = reader.pythonPickle.data
		else:
			data = await data.download.asnc(reader.pythonPickle.bigData)
			
		return pickle.loads(data)
	
	if which == "nested":
		return await _interpret(reader.nested, memoDict)
	
	if which == "pythonNone":
		return None
	
	if which == "pythonEllipsis":
		return Ellipsis
	
	if which == "pythonNotImplemented":
		return NotImplemented
	
	if which == "pythonGlobal":
		globalInfo = reader.pythonGlobal
				
		key = (str(globalInfo.mod), str(globalInfo.name))
		if key not in _globalsDict:
			assert pickleEnabled(), f"""
			Loading of this object requires the usage of the global python object {mod}.{name}.
			This object has not been registered with FusionSC's serialization engine, therefore
			the object you are writing can only be read when full pickle compatibility
			is enabled.
			
			ENABLING THAT MODE IS A SECURITY RISK AS FILES CAN REQUEST ARBITRARY CODE DO BE EXECUTED.
			
			Preferably, please register {mod}.{name} with the serialization engine using the functions
			"cls", "var", or "func" in this module. This will whitelist them for deserialization as well.
			
			If this is not possible, you can enable full pickle compatibility by running inside a
			"with fusionsc.serialize.allowPickle():" statement.
			
			Please keep in mind that no matter which path you take, the person reading this object needs
			to take a similar path as well"""
			
			mod, name = key
			
			import importlib
			mod = importlib.__import__(mod)
			
			result = mod
			for n in name.split("."):
				result = getattr(result, n)
			
			return result
		
		return _globalsDict[key]
		
	if which == "pythonObject":
		with _setLoadMode("strict"):
			objInfo = reader.pythonObject
		 	
			cb = objInfo.createBy
			if cb.which_() == "call":
				func = await _load(cb.call.func, memoDict)
				_checkCallable(func)
				
				args = [await _load(x, memoDict) for x in cb.call.args]
				
				obj = func(*args)
			elif cb.which_() == "newobj":
				cls = await _load(cb.newobj.cls, memoDict)
				_checkCallable(cls)
				
				args = [await _load(x, memoDict) for x in cb.newobj.args]
				
				kwargs = {}
				
				for e in cb.newobj.kwargs:
					k = await _load(e.key, memoDict)
					v = await _load(e.value, memoDict)
					
					kwargs[k] = v
				
				obj = cls.__new__(cls, *args, **kwargs)
			else:
				assert False, "Unknown object creation mode"

			# From here on object may contain itself
			if reader.memoKey != 0:
				assert reader.memoKey not in memoDict
				memoDict[reader.memoKey] = obj
			
			for e in objInfo.listItems:
				obj.append(await _load(e, memoDict))

			for e in objInfo.dictItems:
				k = await _load(e.key, memoDict)
				v = await _load(e.value, memoDict)
				
				obj[k] = v

			s = objInfo.state

			if s.which_() == "setState":
				state = await _load(s.setState, memoDict)
				
				if hasattr(obj, "__setstate__"):
					obj.__setstate__(await _load(s.simple, memoDict))
				else:
					# Default route for state restoration
					
					# State can be either a dict or a tuple(dict, dict)
					# In the later case, the latter is for __slots__
					if isinstance(state, tuple) and len(state) == 2:
						state, slotState = state
					else:
						slotState = None
					
					# Restore state dict
					if state:
						instanceDict = obj.__dict__
						for k, v in state.items():
							obj.__dict__[k] = v
					
					# Restore slot state
					if slotState:
						for k, v in slotState.items():
							setattr(obj, k, v)
				
			elif s.which_() == "withSetter":
				ws = s.withSetter
				
				setter = await _load(ws.setter, memoDict)
				_checkCallable(setter)
				
				val = await _load(ws.value, memoDict)
				setter(obj, val)
			elif w.which_() == "noState":
				pass
			else:
				assert False, "Unknown state setting method"
			
			return obj
	
	if which == "unknownObject":
		return await _load(reader.unknownObject, {"unknown" : None})
	
	if which == "pythonWrapper":
		k = reader.pythonWrapper.which_()
		
		if _loadMode.get() == "fast":
			return await _load(reader.pythonWrapper.wrapped, memoDict)
		
		if k == "mutableArray":
			wrapped = await _load(reader.pythonWrapper.wrapped, memoDict)
			
			assert isinstance(wrapped, np.ndarray)
			return wrapped.copy()
		elif k == "bytes":
			return bytes(await _load(reader.pythonWrapper.wrapped, memoDict))
		elif k == "bytearray":
			return bytearray(await _load(reader.pythonWrapper.wrapped, memoDict))
		elif k == "pickleBuffer":
			return pickle.PickleBuffer(await _load(reader.pythonWrapper.wrapped, memoDict))
			
	import warnings
	warnings.warn(
		"""An unknown type of dynamic object was encountered. This usually means that
		the data being loaded were saved by a newer version of this library. The
		'UnknownObject' returned contains all information stored in this section of
		the message and can still be saved (the data will be written back as encountered
		here"""
	)
	
	return UnknownObject(reader)
