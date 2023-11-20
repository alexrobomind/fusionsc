from ..native import loader
import importlib_resources as ires
import sys

def _createModule(name):
	import importlib
	spec = importlib.machinery.ModuleSpec(name,None)
	return importlib.util.module_from_spec(spec)

w7x = _createModule(f"{__name__}.w7x")
loader.parseSchema("/fusionsc/devices/w7x.capnp", w7x)

jtext = _createModule(f"{__name__}.jtext")
loader.parseSchema("/fusionsc/devices/jtext.capnp", jtext)