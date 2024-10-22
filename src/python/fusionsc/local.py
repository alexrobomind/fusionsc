"""Gives access to native resources of the fusionsc library"""
import typing

from . import native

Capsule = typing.NewType("Capsule", object)

def getStore() -> Capsule:
	return native.local.getStore()

LocalServer = native.local.LocalServer

__all__ = ["getStore", "LocalServer"]
