"""
This module contains the auto-configuration mechanism for fusionsc.
"""

from . import structio
from . import asnc

import pathlib
import traceback
import contextvars
import os

userPath = pathlib.Path.home() / ".fusionsc.yaml"

_envName = "FUSIONSC_CONFIG_PATH"
if _envName in os.environ:
	userPath = pathlib.Path(os.environ[_envName])

config = {}
siteName = None

_applied = contextvars.ContextVar("fusionsc.config._applied", default = False)
_configContext = asnc.EventLoopLocal()

def context():
	"""
	Provides a context where the library has been properly configured. Other modules
	use this context to look up their default values.
	"""
	if _configContext.isSet():
		return _configContext.value
	
	# Start with an empty context
	ctx = contextvars.Context()
	_configContext.value = ctx
	
	# Configure the contents of the context
	ctx.run(_apply)
	
	return ctx

def _apply():
	from . import resolve
	from . import backends
	from . import remote
	
	# Apply default resolvers first
	resolve._addDefaults()
		
	try:
		if "backend" in config:
			backends.alwaysUseBackend(remote.connect(str(config["backend"])))
		
		if "resolve" in config:
			for entry in config["resolve"]:
				if "warehouse" in entry:
					resolve.connectWarehouse(entry["warehouse"])
				if "archive" in entry:
					resolve.importOfflineData(entry["archive"])
		
		if "w7x" in config:
			if "defaultCoilsUrl" in config["w7x"]:
				from .devices import w7x
				w7x._loadDefaultCoils(config["w7x"]["defaultCoilsUrl"])
	except:
		print("!!! --- Failed to apply user configuration --- !!!")
		print("")
		print("More information about the error follows:")
		print("")
		traceback.print_exc()

if userPath.exists():
	try:
		with userPath.open("rb") as f:
			config = structio.load(f, lang="yaml")
	
		if "siteName" in config:
			siteName = str(config["siteName"])
			
	except:
		print("!!! --- Failed to load user configuration --- !!!")
		print("")
		print("More information about the error follows:")
		print("")
		traceback.print_exc()
	
defaults = {
	"ipp-hgw" : {
		"siteName" : "ipp-hgw",
		"backend" : "http://fusionsc-site:8888/load-balancer",
		"resolve" : [
			{"warehouse" : "remote:w7xdb"}
		],
		"w7x" : {
			"defaultCoilsUrl" : "remote:w7xdb#precomputedCoils/cadCoils"
		}
	}
}

def configCli():
	import argparse
	import sys
	
	global config
	
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="command")
	
	defaultParser = subparsers.add_parser("default")
	defaultParser.add_argument("name", choices = list(defaults))
	
	resetParser = subparsers.add_parser("reset")
	showParser = subparsers.add_parser("show")
	
	setBackendParser = subparsers.add_parser("set-backend")
	setBackendParser.add_argument("backend")
	
	resetBackendParser = subparsers.add_parser("reset-backend")
	
	resolveParser = subparsers.add_parser("resolve")
	resolveSubparsers = resolveParser.add_subparsers(dest="resolveCommand")
	
	resolveAdd = resolveSubparsers.add_parser("add")
	resolveAdd.add_argument("--type", dest="fileType", default=None, choices = ["warehouse", "archive"])
	resolveAdd.add_argument("urlOrPath")
	
	resolveRemove = resolveSubparsers.add_parser("remove")
	resolveRemove.add_argument("expression")
	
	args = parser.parse_args()
	
	if args.command == "default":
		if args.name not in defaults:
			print(f"Requested default {args.name} not found")
			sys.exit(1)
		
		print(f"Loading default '{args.name}' ...")
		config = defaults[args.name]
	
	if args.command == "reset":
		print(f"Deleting {userPath} ...")
		userPath.unlink(missing_ok = True)
		sys.exit(0)
	
	if args.command == "show":
		print(f"Configuration at {userPath}:")
		print(structio.dumps(config, lang = "yaml"))
		sys.exit(0)
	
	if args.command == "set-backend":
		config["backend"] = args.backend
	
	if args.command == "reset-backend":
		del config["backend"]
	
	if args.command == "resolve":
		if "resolve" not in config:
			config["resolve"] = []
			
		if args.resolveCommand == "add":
			url = args.urlOrPath
			
			fileType = args.fileType
			
			if fileType is None:
				try:
					# Check if URL or path
					isUrl = url.startswith("http:") or url.startswith("sqlite:") or url.startswith("ws:")
					
					# If we are a path, open it to check contents
					if isUrl:
						fileType = "warehouse"
					else:
						with pathlib.Path(url).open("rb") as f:
							magicBytes = f.read(7)
						
						if magicBytes == b"FSCARCH":
							fileType = "archive"
						elif magicBytes == b"SQLite ":
							fileType = "warehouse"
						else:
							raise ValueError("Invalid file contents")
				except:
					print("Could not determine file type from contents. Please specify with the --type {warehouse / archive} argument.")
					print()
					print("More information about the error follows:")
					print("")
					traceback.print_exc()
					
					sys.exit(1)
			
			if fileType == "warehouse":
				import urllib.parse
				parsed = urllib.parse.urlparse(url, scheme = "sqlite")
				
				if parsed.scheme == "sqlite":
					path = pathlib.Path(parsed.path)
					parsed = parsed._replace(path = str(path.absolute()).replace("\\", "/"), netloc="abs")
				
				url = urllib.parse.urlunparse(parsed)
			else:
				path = pathlib.Path(url)
				url = str(path.absolute())
				
			print(f"Adding entry {fileType}: {url} ...")
			config["resolve"].append({fileType : url})
		
		if args.resolveCommand == "remove":
			print(f"Removing entries matching '{args.expression}' ...")
			
			import fnmatch
			
			config["resolve"] = [
				val
				for val in config["resolve"]
				if not fnmatch.fnmatch(list(val.values())[0], args.expression)
			]
		
	if "resolve" in config and len(config["resolve"]) == 0:
		del config["resolve"]
			
	print("New config:\n")
	print(structio.dumps(config, lang = "yaml"))
	
	print()
	print(f"Writing config to {userPath} ...")
	with userPath.open("w") as f:
		structio.dump(config, f, lang = "yaml")

if __name__ == "__main__":
	configCli()
