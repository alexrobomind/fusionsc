#pragma once

#include <stdint.h>

/**
 * \defgroup capi_store C interface for data store.
 * @{
 *
 * C interface for FusionSC's cross-thread data store.
 *
 * \note
 * This interface is not designed for manual use. Instead, users should pass pointers
 * to fusionsc_DataStore into the library's startup options to link multiple FusionSC
 * instances together.
 */

#ifdef __cplusplus
extern "C" {
#endif

struct fusionsc_DataStore;
struct fusionsc_DataStoreEntry;
struct fusionsc_DataHandle;

//! Data store object
struct fusionsc_DataStore {
	uint16_t version = 1;
	
	void (*incRef)(fusionsc_DataStore*);
	void (*decRef)(fusionsc_DataStore*);
	
	fusionsc_DataStoreEntry* (*publish)(fusionsc_DataStore*, const unsigned char*, size_t, fusionsc_DataHandle*);
	fusionsc_DataStoreEntry* (*query)(fusionsc_DataStore*, const unsigned char*, size_t);
	
	void (*gc)(fusionsc_DataStore*);
};

//! Handle to a binary blob with custom deleter.
struct fusionsc_DataHandle {
	const unsigned char* dataPtr;
	size_t dataSize;
	
	void (*free)(fusionsc_DataHandle* hdl);
};

//! Data store row handle
struct fusionsc_DataStoreEntry {
	uint16_t version = 1;
	
	const unsigned char* dataPtr;
	const unsigned char* keyPtr;
	size_t keySize;
	size_t dataSize;
	
	void (*incRef)(fusionsc_DataStoreEntry*);
	void (*decRef)(fusionsc_DataStoreEntry*);
};

#ifdef __cplusplus
}
#endif
