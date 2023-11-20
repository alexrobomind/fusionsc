from . import capnp

from ..native import loader
import importlib_resources as ires
import sys

loader.roots["fusionsc"] = ires.files().joinpath("fusionsc")

files = [
	"data.capnp",
	"data-archive.capnp",
	"dynamic.capnp",
	"flt.capnp",
	"geometry.capnp",
	"hfcam.capnp",
	"hint.capnp",
	"http.capnp",
	"index.capnp",
	"jobs.capnp",
	"magnetics.capnp",
	"matcher.capnp",
	"networking.capnp",
	"offline.capnp",
	"random.capnp",
	"services.capnp",
	"streams.capnp",
	"vmec.capnp",
	"warehouse.capnp"
]

thisModule = sys.modules[__name__]
for file in files:
	loader.parseSchema(f"/fusionsc/{file}", thisModule)

del thisModule
del file
del files

from . import devices