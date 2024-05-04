"""
This module contains the auto-configuration mechanism for fusionsc.
"""

from . import structio

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

def apply():
	from . import resolve
	from . import backends
	from . import remote
	
	if _applied.get():
		return
		
	if "backend" in config:
		backends.alwaysUseBackend(remote.connect(str(config["backend"])))
	
	if "resolve" in config:
		for entry in config["resolve"]:
			if "warehouse" in entry:
				resolve.connectWarehouse(entry["warehouse"])
			if "archive" in entry:
				resolve.importOfflineData(entry["archive"])
	
	_applied.set(True)

if userPath.exists():
	try:
		with userPath.open() as f:
			config = structio.load(f, lang="yaml")
	
		if "siteName" in config:
			siteName = str(config["siteName"])
				
		apply()
	except:
		print("!!! --- Failed to apply user configuration --- !!!")
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
		]
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
	resolveAdd.add_argument("type", choices = ["warehouse", "archive"])
	resolveAdd.add_argument("url")
	
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
			if args.type == "archive":
				path = pathlib.Path(args.url)
				url = str(path.absolute())
			else:
				import urllib.parse
				parsed = urllib.parse.urlparse(args.url, scheme = "sqlite")
				
				if parsed.scheme == "sqlite":
					path = pathlib.Path(parsed.path)
					parsed = parsed._replace(path = str(path.absolute()).replace("\\", "/"), netloc="abs")
				
				url = urllib.parse.urlunparse(parsed)
			print(f"Adding entry {args.type}: {url} ...")
			config["resolve"].append({args.type : url})
		
		if args.resolveCommand == "remove":
			print(f"Removing entries matching '{args.expression}' ...")
			
			import re
			expr = re.compile(args.expression)
			
			config["resolve"] = [
				val
				for val in config["resolve"]
				if not expr.match(list(val.values())[0])
			]
		
	if "resolve" in config and len(config["resolve"]) == 0:
		del config["resolve"]
			
	print("New config:\n")
	print(structio.dumps(config, lang = "yaml"))
	
	print()
	print(f"Writing config to {userPath} ...")
	with userPath.open("w") as f:
		structio.dump(config, f, lang = "yaml")