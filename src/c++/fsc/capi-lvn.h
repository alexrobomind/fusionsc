#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// It can become difficult to form message queues between differently linked / compiled
// version of this library (which becomes a problem when it is used in Python projects).
// To mitigate this, the LocalVatNetwork class relies under the hood on a set of C structs
// with virtual functions.
//
// When libaries want to exchange connections beyond their C++ ABI compatibilty scope
// (e.g. different library version, libraries built against different versions of Cap'n'proto),
// they can exchange instances of these types instead of the wrapping C++ types to overcome
// the ABI incompatibility.

struct fusionsc_LvnHub;
struct fusionsc_LvnListener;
struct fusionsc_Lvn;
struct fusionsc_LvnEndPoint;
struct fusionsc_LvnMessage;

/**
 * This type represents the actual local network. Users can join the network by calling its
 * 'join' method.
 */
struct fusionsc_LvnHub {
	uint16_t version = 1;
	
	void (*incRef)(fusionsc_LvnHub*);
	void (*decRef)(fusionsc_LvnHub*);
	fusionsc_Lvn* (*join)(fusionsc_LvnHub*, fusionsc_LvnListener*);
};

/**
 * This interface needs to be implemented by users that wish to join a local vat network.
 */
struct fusionsc_LvnListener {
	uint16_t version = 1;

	//! Called by the hub when this listener is no longer used.	
	void (*free)(fusionsc_LvnListener*);
	
	/**
	 * Create an instance of fusionsc_LvnEndPoint by accepting an incoming remote endpoint.
	 * Arguments are the address of the remote LVN instance opening the connection, and
	 * an owning pointer to a remote comms endpoint that can be used to push messages to
	 * the other end.
	 *
	 * Must return an owning pointer to the local end of the connection.
	 */
	fusionsc_LvnEndPoint* (*accept)(fusionsc_LvnListener*, uint64_t, fusionsc_LvnEndPoint*);
};

//! A network endpoint can connect to remote endpoints.
struct fusionsc_Lvn {	
	uint16_t version = 1;
	
	uint64_t address;
	
	void (*incRef)(fusionsc_Lvn*); // Increases the reference count of this object
	void (*decRef)(fusionsc_Lvn*); // Decreases the reference count of this object
	
	fusionsc_LvnEndPoint* (*connect)(fusionsc_Lvn*, uint64_t, fusionsc_LvnEndPoint*); // Creates a connection to the remote end
};

/**
 * This class represents a connection endpoint that the remote side of the connection can pass
 * messages to. Connections are handshaked by passing an instance of this type to the other
 * side while simultaneously receiving one from there.
 */
struct fusionsc_LvnEndPoint {
	uint16_t version = 1;
	
	void (*incRef)(fusionsc_LvnEndPoint*); // Increases the reference count of this object
	void (*decRef)(fusionsc_LvnEndPoint*); // Decreases the reference count of this object
	
	void (*close)(fusionsc_LvnEndPoint*); // Notifies the endpoint that no more messages will be sent
	void (*receive)(fusionsc_LvnEndPoint*, fusionsc_LvnMessage*); // Return 0 if message was accepted, 1 if it was discarded
};

//! Represents a segment owned by a Cap'n'proto message.
struct fusionsc_SegmentInfo {
	const void* data;
	size_t sizeInWords;
};

/**
 * Represents an incoming Cap'n'proto messages with attached file descriptors.
 *
 * WARNING: The lifetime of the file descriptors is not tied to the lifetime of this message.
 * Users must take care of closing the file descriptors even when they call the free method
 * of this type.
 */
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
