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
	'readArchive', 'download', 'writeArchive'
]

@asyncFunction
def download(ref):
	return downloadAsync(ref)

@asyncFunction
def writeArchive(data, filename: str):
	return writeArchiveAsync(data, filename)

@asyncFunction
def readArchive(filename: str):
	return download.asnc(readArchiveAsync(filename))
