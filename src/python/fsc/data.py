from .native.data import (
	download,
	publish,
	
	openArchive,
	writeArchive
)

from .asnc import eager

__all__ = [
	# Imported from native
	'download', 'publish', 'openArchive', 'writeArchive',
	
	# Locally defined
	'readArchive'
]

def readArchive(filename: str):
	return download(openArchive(filename))
