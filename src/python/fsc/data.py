from . import native
from .asnc import asyncFunction

@asyncFunction
def publish(data):
	return native.data.publish(ref)

@asyncFunction
def download(ref):
	return native.data.downloadAsync(ref)

@asyncFunction
def writeArchive(data, filename: str):
	return native.data.writeArchiveAsync(data, filename)

@asyncFunction
def readArchive(filename: str):
	return download.asnc(native.data.openArchive(filename))
