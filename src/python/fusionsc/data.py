from . import native
from . import inProcess
from .asnc import asyncFunction

@asyncFunction
def openArchive(filename: str):
	return inProcess.localResources().openArchive(filename)

def publish(data):
	inThreadRef = native.data.publish(ref)
	cloneResult = inProcess.localResources().download(inThreadRef)
	return cloneResult.ref

@asyncFunction
def download(ref):
	return native.data.downloadAsync(ref)

@asyncFunction
def writeArchive(data, filename: str):
	return inProcess.localResources().writeArchive(filename, data)

@asyncFunction
def readArchive(filename: str):
	return download.asnc(openArchive(filename))
