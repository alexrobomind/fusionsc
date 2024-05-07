"""
This module can be used to perform import and export operations between python objects
(dicts, lists, data readers / builders, numpy arrays) and self-describing nested data
formats (currently JSON, YAML, CBOR, BSON, MSGPACK, and UBJSON).
"""

from . import native
from .asnc import asyncFunction

from typing import Union

_langs = {
	'json' : native.formats.Language.JSON,
	'yaml' : native.formats.Language.YAML,
	'cbor' : native.formats.Language.CBOR,
	'bson' : native.formats.Language.BSON,
	'ubjson' : native.formats.Language.UBJSON,
	'msgpack' : native.formats.Language.MSGPACK
}

def _checkLang(lang):
	assert lang in _langs, f"Language must be one of {list(_langs)}"
	return _langs[lang]

def dumps(data, lang='json', compact=False, binary=None) -> Union[str, bytes]:
	"""
	Write the object into a bytes or str representation according to 'lang'.
	
	Arguments:
		- data: Target object to serialize
		- lang: Name of the output language (must be json, yaml, cbor, msgpack, ubjson, or bson)
		- compact: Whether to write outputs in a short-handed notation where possible.
		- binary; Whether to return the result as bytes instead of str (default is False for string
		  languages and True for binary languages)
	
	Returns:
		A bytes or str object holding the stored data according to the requested format.
	"""
	asBytes = native.formats.dumpToBytes(data, _checkLang(lang), compact)
	
	if binary is None:
		binary = (lang not in ["json", "yaml"])
	
	if binary:
		return asBytes
	
	return asBytes.decode()

def dump(data, file, lang='json', compact=False):
	"""
	Write the object into a file object. The object needs to have a fileno method
	returning a C file descriptor.
	
	Arguments:
		- data: Target object to serialize
		- file: File-like object
		- lang: Name of the output language (must be json, yaml, cbor, msgpack, ubjson, or bson)
		- compact: Whether to write outputs in a short-handed notation where possible.
	"""
	file.flush()
	fd = file.fileno()
	native.formats.dumpToFd(data, fd, _checkLang(lang), compact)

@asyncFunction
async def recursiveDumps(data, lang='json', compact=False, binary=None) -> Union[str, bytes]:
	"""
	Like dumps(...), but also serializes nested DataRefs.
	"""
	asBytes = await native.formats.dumpAllToBytes(data, _checkLang(lang), compact)
	
	if binary is None:
		binary = (lang not in ["json", "yaml"])
	
	if binary:
		return asBytes
	
	return asBytes.decode()

@asyncFunction
async def recursiveDump(data, file, lang='json', compact=False):
	"""
	Like dump(...), but also serializes nested DataRefs.
	"""
	file.flush()
	fd = file.fileno()
	await native.formats.dumpAllToFd(data, fd, _checkLang(lang), compact)

def load(src, dst=None, lang='json'):
	"""
	Load the formatted data. Can either deserialize the input as a nested structure of
	bytes, str, dict, and list, or alternatively deserialize INTO a target object (returning
	the result in either case).
	
	Deserializing into a target object allows a large amount of conversions based on the
	destination types (shorthand notations for struct, decoding base64-encoded binary data, etc.)
	and reduces the amount of intermediate copies required. Target objects can be nested structures
	of dicts, lists, and struct builders.
	
	Arguments:
		- src: file (with fileno. attribute), str, or bytes-like object serving as data source
		- dst: Optional target to deserialize into
		- lang: Language in which the source data is stored (must be json, yaml, cbor, msgpack, ubjson, or bson)
	
	Returns:
		- The deserialized data. If "dst" is specified, this same object is returned.
	"""
	cl = _checkLang(lang)
	
	if hasattr(src, 'fileno'):
		return native.formats.readFd(src.fileno(), dst, cl)
	
	if isinstance(src, str):
		return native.formats.readBuffer(src.encode(), dst, cl)
	
	return native.formats.readBuffer(src, dst, cl)