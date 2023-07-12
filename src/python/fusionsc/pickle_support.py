"""
This module contains functions that are required to unpickle stored files.
"""

from . import native

def unpickleReader(pickleVersion, version, data):
	return native.capnp._unpickleReader(pickleVersion, version, data)

def unpickleBuilder(pickleVersion, version, data):
	return native.capnp._unpickleBuilder(pickleVersion, version, data)

def unpickleRef(pickleVersion, version, data):
	return native.capnp._unpickleRef(pickleVersion, version, data)