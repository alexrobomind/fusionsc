from .native.data import (
	downloadAsync,
	publish,
	
	openArchive,
	writeArchiveAsync
)

from .asnc import eager

__all__ = [
	# Imported from native
	'downloadAsync', 'publish', 'openArchive', 'writeArchiveAsync',
	
	# Locally defined
	'readArchive', 'readArchiveAsync', 'download', 'writeArchive'
]

def download(ref):
	return downloadAsync(ref).wait()

def writeArchive(data, filename: str):
	return writeArchiveAsync(data, filename).wait()

def readArchive(filename: str):
	return readArchiveAsync(filename).wait()

def readArchiveAsync(filename: str):
	return downloadAsync(openArchive(filename))