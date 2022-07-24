from .native.data import (
	download,
	publish,
	
	openArchive,
	writeArchive
)

from .asnc import Promise

__all__ = [
	# Imported from native
	'download', 'publish', 'openArchive', 'writeArchive',
	
	# Locally defined
	'readArchive'
]

def readArchive(filename: str) -> Promise:
	return download(openArchive(str))
