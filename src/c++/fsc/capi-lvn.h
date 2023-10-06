#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Note: This namespace is experimental.

// It can become difficult to form message queues between differently linked / compiled
// version of this library (which becomes a problem when it is used in Python projects).
// To mitigate this, the local vat network will in future switch to an implementation
// based on function pointers and standard layout types.

struct fusionsc_LvnHub;
struct fusionsc_LvnListener;
struct fusionsc_Lvn;
struct fusionsc_LvnEndPoint;
struct fusionsc_LvnMessage;

struct fusionsc_LvnHub {
	uint16_t version = 1;
	
	void (*incRef)(fusionsc_LvnHub*);
	void (*decRef)(fusionsc_LvnHub*);
	fusionsc_Lvn* (*join)(fusionsc_LvnHub*, fusionsc_LvnListener*);
};

struct fusionsc_LvnListener {
	uint16_t version = 1;
	
	void (*free)(fusionsc_LvnListener*);
	fusionsc_LvnEndPoint* (*accept)(fusionsc_LvnListener*, uint64_t, fusionsc_LvnEndPoint*);
};

struct fusionsc_Lvn {	
	uint16_t version = 1;
	
	uint64_t address;
	
	void (*incRef)(fusionsc_Lvn*); // Increases the reference count of this object
	void (*decRef)(fusionsc_Lvn*); // Decreases the reference count of this object
	
	fusionsc_LvnEndPoint* (*connect)(fusionsc_Lvn*, uint64_t, fusionsc_LvnEndPoint*); // Creates a connection to the remote end
};

struct fusionsc_LvnEndPoint {
	uint16_t version = 1;
	
	void (*incRef)(fusionsc_LvnEndPoint*); // Increases the reference count of this object
	void (*decRef)(fusionsc_LvnEndPoint*); // Decreases the reference count of this object
	
	void (*close)(fusionsc_LvnEndPoint*); // Notifies the endpoint that no more messages will be sent
	void (*receive)(fusionsc_LvnEndPoint*, fusionsc_LvnMessage*); // Return 0 if message was accepted, 1 if it was discarded
};

struct fusionsc_SegmentInfo {
	const void* data;
	size_t sizeInWords;
};

struct fusionsc_LvnMessage {
	fusionsc_SegmentInfo* segmentInfo;
	size_t segmentCount;
	
	int* fds;
	size_t fdCount;
	
	void (*free)(fusionsc_LvnMessage*);
};

#ifdef __cplusplus
}
#endif