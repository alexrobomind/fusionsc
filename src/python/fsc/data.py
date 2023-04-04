from . import native
from . import worker
from .asnc import asyncFunction

@asyncFunction
def openArchive(filename: str):
	return worker.localResources().openArchive(filename)

def publish(data):
	inThreadRef = native.data.publish(ref)
	cloneResult = worker.localResources().download(inThreadRef)
	return cloneResult.ref

@asyncFunction
def download(ref):
	return native.data.downloadAsync(ref)

@asyncFunction
def writeArchive(data, filename: str):
	return worker.localResources().writeArchive(filename, data)

@asyncFunction
def readArchive(filename: str):
	return download.asnc(openArchive(filename))
