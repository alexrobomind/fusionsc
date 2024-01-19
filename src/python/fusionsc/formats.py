from . import native

from .asnc import asyncFunction

"""
This module can be used to perform import and export operations between python objects
(dicts, lists, data readers / builders, numpy arrays) and self-describing nested data
formats (currently JSON, YAML, CBOR, and BSON).
"""

_langs = {
	'json' : native.formats.Language.JSON,
	'yaml' : native.formats.Language.YAML,
	'cbor' : native.formats.Language.CBOR,
	'bson' : native.formats.Language.BSON
}

def _checkLang(lang):
	assert lang in _langs, f"Language must be one of {list(_langs)}"
	return _langs[lang]

def dumps(data, lang='json', compact=False, binary=False):
	asBytes = native.formats.dumpToBytes(data, _checkLang(lang), compact)
	
	if binary:
		return asBytes
	
	return asBytes.decode()

def dump(data, file, lang='json', compact=False):
	fd = file.fileno()
	native.formats.dumpToFd(data, fd, _checkLang(lang), compact)

@asyncFunction
async def recursiveDumps(data, lang='json', comapct=False, binary=Falase):
	asBytes = await native.formats.dumpAllToBytes(data, _checkLang(lanc), compact)
	
	if binary:
		return asBytes
	
	return asBytes.decode()

@asyncFunction
async def recursiveDump(data, file, lang='json', compact=False):
	fd = file.fileno()
	await native.formats.dumpAllToFd(data, fd, _checkLang(lanc), compact)

def read(src, dst=None, lang='json'):
	cl = _checkLang(lang)
	
	if hasattr(src, 'fileno'):
		return native.formats.readFd(src.fileno(), dst, cl)
	
	if isinstance(src, str):
		return native.formats.readBuffer(src.encode(), dst, cl)
	
	return native.formats.readBuffer(src, dst, cl)