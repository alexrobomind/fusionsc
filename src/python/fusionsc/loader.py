"""
Creates python accessors for native .capnp files
"""

from .native import loader as nativeLoader
from typing import Optional

roots: dict = nativeLoader.roots

def getType(id: int):
	return nativeLoader.getType(id)

def parseSchema(path: str, scope: object, pathRoot: Optional[object] = None):
	if isinstance(scope, str):
		# We just got a module name and need to create the corresponding modules
		# first.
		import sys
		import importlib
		
		components = scope.split(".")
		
		# Find the deepest potential parent module
		parentIdx = 0
		parent = None
		
		for i in range(1, len(components) + 1):
			c = components[i-1]
			candidate = ".".join(components[:i])
			
			if candidate in sys.modules:
				parentIdx = i
				parent = sys.modules[candidate]
			
			if hasattr(parent, c):
				parentIdx = i
				parent = getattr(parent, c)
		
		# Add the remaining child modules
		for i in range(parentIdx + 1, len(components) + 1):
			moduleName = ".".join(components[:i])
			
			spec = importlib.machinery.ModuleSpec(f"{moduleName}",None)
			mod = importlib.util.module_from_spec(spec)
			sys.modules[moduleName] = mod
			
			if parent is not None:
				setattr(parent, components[i-1], mod)
			
			parent = mod
		
		scope = parent
		
	return nativeLoader.parseSchema(path, scope, pathRoot)
