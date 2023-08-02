"""
This module contains functions that are required to unpickle stored files.
"""

from . import native

def unpickleReader(version, data):
	return native.capnp._unpickleReader(version, data)

def unpickleBuilder(version, data):
	return native.capnp._unpickleBuilder(version, data)

def unpickleRef(version, data):
	return native.capnp._unpickleRef(version, data)

def unpickleEnum(version, typeInfo, enumerant):
	return native.capnp._unpickleEnum(version, typeInfo, enumerant)