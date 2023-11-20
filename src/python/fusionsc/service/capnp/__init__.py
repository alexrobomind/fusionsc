from ...native import loader
import importlib_resources as ires
import sys

thisModule = sys.modules[__name__]

loader.roots["capnp"] = ires.files()

loader.parseSchema("/capnp/c++.capnp", thisModule)
loader.parseSchema("/capnp/schema.capnp", thisModule)
loader.parseSchema("/capnp/rpc.capnp", thisModule)

del thisModule