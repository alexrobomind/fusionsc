from . import data
from . import config

from . import warehouse
from . import wrappers


def timestamp():
	import datetime
	return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Session ID
_session = timestamp()

# Destination for logging
def _getLogDst():
	logCfg = config.config.get("logging", {})
	dst = logCfg.get("destination", None)
	
	return dst

logDestination = _getLogDst()

def shouldLog(module):
	if logDestination is None:
		return False
	
	logCfg = config.config.get("logging", {})
	modules = logCfg.get("modules", [])
	
	return module in modules

@wrappers.asyncFunction
async def log(module, path, data):
	if not shouldLog(module):
		return
	
	dstPath = f"{_session}/{module}/{path}"
	
	wh = await warehouse.open.asnc(logDestination)
	await wh.put.asnc(dstPath, data)
	
	print(f"Log {dstPath}")