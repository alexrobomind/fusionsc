"""
This module parses the Cap'n'proto schema files and exposes them as python classes

The following Cap'n'proto files are converted:

- fusionsc.service: `FusionSC specific service <https://github.com/alexrobomind/fusionsc/tree/main/src/python/fusionsc/serviceDefs/fusionsc>`_
- fusionsc.service.capnp: `Cap'n'proto builtins <https://github.com/alexrobomind/fusionsc/tree/main/src/python/fusionsc/serviceDefs/capnp>`_
- fusionsc.service.devices: `Device-specific service classes <https://github.com/alexrobomind/fusionsc/tree/main/src/python/fusionsc/serviceDefs/fusionsc/devices>`_
"""
from . import loader

import importlib_resources as ires
import sys

_files = ires.files()

_capnpFiles = _files.joinpath("serviceDefs", "capnp")
_fscFiles = _files.joinpath("serviceDefs", "fusionsc")

# Add new root entries to the loader
loader.roots["capnp"] = _capnpFiles
loader.roots["fusionsc"] = _fscFiles

for _file in _capnpFiles.iterdir():
	loader.parseSchema(f"/capnp/{_file.name}", "fusionsc.service.capnp")

for _file in _fscFiles.iterdir():
	if "test" in _file.name or "internal" in _file.name:
		continue
	if ".capnp" not in _file.name:
		continue
	
	loader.parseSchema(f"/fusionsc/{_file.name}", "fusionsc.service")
	
for _device in ["w7x", "jtext"]:
	loader.parseSchema(f"/fusionsc/devices/{_device}.capnp", f"fusionsc.service.devices.{_device}")
