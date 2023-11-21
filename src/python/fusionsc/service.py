from . import capnp

from .native import loader
import importlib_resources as ires
import sys

_files = ires.files()

_capnpFiles = _files.joinpath("serviceDefs", "capnp")
_fscFiles = _files.joinpath("serviceDefs", "fusionsc")

def _createModule(name):
	import importlib
	spec = importlib.machinery.ModuleSpec(f"{__name__}.{name}",None)
	return importlib.util.module_from_spec(spec)

# Populate capnp module
loader.roots["capnp"] = _capnpFiles
capnp = _createModule("capnp")
for _file in _capnpFiles.iterdir():
	loader.parseSchema(f"/capnp/{_file.name}", capnp)

# Populate fusionsc module
loader.roots["fusionsc"] = _fscFiles
_thisModule = sys.modules[__name__]
for _file in _fscFiles.iterdir():
	if "test" in _file.name or "internal" in _file.name:
		continue
	if ".capnp" not in _file.name:
		continue
	loader.parseSchema(f"/fusionsc/{_file.name}", _thisModule)

# Populate device submodules
devices = _createModule("devices")
for _device in ["w7x", "jtext"]:
	_submodule = _createModule(f"devices.{_device}")
	setattr(devices, _device, _submodule)
	loader.parseSchema(f"/fusionsc/devices/{_device}.capnp", _submodule)