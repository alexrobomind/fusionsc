#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Note: This namespace is experimental.

// It can become difficult to form message queues between differently linked / compiled
// version of this library (which becomes a problem when it is used in Python projects).
// To mitigate this, the local vat network will in future switch to an implementation
// based on function pointers and standard layout types.

struct fusionsc_LvnHub;
struct fusionsc_LvnHub_VTable;
struct fusionsc_LvnListener;
struct fusionsc_LvnListener_VTable;
struct fusionsc_Lvn;
struct fusionsc_Lvn_VTable;
struct fusionsc_LvnEndPoint;
struct fusionsc_LvnEndPoint_VTable;
struct fusionsc_LvnMessage;
struct fusionsc_LvnMessage_VTable;

struct fusionsc_LvnHub {
	fusionsc_LvnHub_VTable* v;
};

struct fusionsc_LvnHub_VTable {
	uint16_t version = 1;
	
	void (*incRef)(fusionsc_LvnHub*);
	void (*decRef)(fusionsc_LvnHub*);
	fusionsc_Lvn* (*join)(fusionsc_LvnHub*, fusionsc_LvnListener*);
};

struct fusionsc_LvnListener {
	fusionsc_LvnListener_VTable* v;
};

struct fusionsc_LvnListener_VTable {
	uint16_t version = 1;
	
	void (*free)(fusionsc_LvnListener*);
	fusionsc_LvnEndPoint* (*accept)(fusionsc_LvnListener*, uint64_t, fusionsc_LvnEndPoint*);
};

struct fusionsc_Lvn {
	fusionsc_Lvn_VTable* v;
	
	const uint64_t address;
};

struct fusionsc_Lvn_VTable {
	uint16_t version = 1;
	
	void (*incRef)(fusionsc_Lvn*); // Increases the reference count of this object
	void (*decRef)(fusionsc_Lvn*); // Decreases the reference count of this object
	fusionsc_LvnEndPoint* (*connect)(fusionsc_Lvn*, uint64_t, fusionsc_LvnEndPoint*); // Creates a connection to the remote end
};

struct fusionsc_LvnEndPoint {
	fusionsc_LvnEndPoint_VTable* v;
};

struct fusionsc_LvnEndPoint_VTable {
	uint16_t version = 1;
	
	void (*free)(fusionsc_LvnEndPoint*);
	uint8_t (*receive)(fusionsc_LvnMessage*); // Return 0 if OK, 1 if connection is closed / failed
};

struct fusionsc_SegmentInfo {
	const void* data;
	size_t sizeInWords;
};

struct fusionsc_FdList {
	int* data;
	size_t size;
};

struct fusionsc_LvnMessage {
	fusionsc_LvnMessage_VTable* v;
};
	
struct fusionsc_LvnMessage_VTable {
	uint16_t version = 1;
	
	void (*free)(fusionsc_LvnMessage*);
	fusionsc_SegmentInfo (*getSegment)(fusionsc_LvnMessage*, uint32_t);
	
	// Gives access to the FD list attached to the message.
	// NOTE: Once called, closing the FDs is responsibility
	// of the caller. Do not call this more than once.
	fusionsc_FdList (*getFds)(fusionsc_LvnMessage*); 
};

#ifdef __cplusplus
}
#endif