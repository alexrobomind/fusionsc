from . import native
from .asnc import asyncFunction

def openArchive(filename: str):
	return native.data.openArchive(filename)

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
	return download.asnc(openArchive(filename))
