#pragma once

#include <stdint.h>
#include <stddef.h>

/**
 * \defgroup capi_lvn C interface for local vat networks.
 * @{
 *	 
 * It can become difficult to form message queues between differently linked / compiled
 * version of this library (which becomes a problem when it is used in Python projects).
 * To mitigate this, the LocalVatNetwork class relies under the hood on a set of C structs
 * with virtual functions.
 *
 * When libaries want to exchange connections beyond their C++ ABI compatibilty scope
 * (e.g. different library version, libraries built against different versions of Cap'n'proto),
 * they can exchange instances of these types instead of the wrapping C++ types to overcome
 * the ABI incompatibility.
 *
 * \note
 * This interface is not really designed to be user friendly, but to minimize surface area.
 * We recommend using the C++ LocalVatNetwork API, which is interconvertible with this
 * interface.
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

struct fusionsc_LvnHub;
struct fusionsc_LvnListener;
struct fusionsc_Lvn;
struct fusionsc_LvnEndPoint;
struct fusionsc_LvnMessage;

//! C interface for the local vat network.
struct fusionsc_LvnHub {
	//! Reserved for protocol evolution.
	uint16_t version = 1;
	
	void (*incRef)(fusionsc_LvnHub*);
	void (*decRef)(fusionsc_LvnHub*);
	
	/** \brief Join the local vat network.
	 * \param self Pointer to this object.
	 * \param listener Listener for incoming connections.
	 * \returns An endpoint for the local vat network.
	 */
	fusionsc_Lvn* (*join)(fusionsc_LvnHub* self, fusionsc_LvnListener* listener);
};

//! This interface needs to be implemented by clients that wish to join a local vat network.
struct fusionsc_LvnListener {
	//! Reserved
	uint16_t version = 1;

	//! Called by the hub when this listener is no longer used.	
	void (*free)(fusionsc_LvnListener*);
	
	/**
	 * \brief Accept a connection request and handshake the connection objects.
	 *
	 * \param self Pointer to this object.
	 * \param remoteAddress Address of the remote endpoint requesting a connection.
	 * \param remoteLink Connection endpoint that this side should use to send messages. Owning pointer,
	 *                   call decRef after use.
	 * \returns A connection endpoint that the remote side should send its messages to for this connection.
	 */
	fusionsc_LvnEndPoint* (*accept)(fusionsc_LvnListener* self, uint64_t remoteAddress, fusionsc_LvnEndPoint* remoteConnection);
};

//! A network endpoint can connect to remote endpoints.
struct fusionsc_Lvn {
	//! Reserved
	uint16_t version = 1;
	
	//! Address of this network endpoint.
	uint64_t address;
	
	void (*incRef)(fusionsc_Lvn*); // Increases the reference count of this object
	void (*decRef)(fusionsc_Lvn*); // Decreases the reference count of this object
	
	/**
	 * \brief Initiate a connection to a remote endpoint.
	 *
	 * \param self Pointer to this object.
	 * \param remoteAddress Destination object to connect to.
	 * \param localEndPoint Prepared connection side that the remote side can pass messages to.
	 * \returns Connection endpoint representing the other side (that this side can push messages to) or nullptr.
	 */
	fusionsc_LvnEndPoint* (*connect)(fusionsc_Lvn*, uint64_t, fusionsc_LvnEndPoint*); // Creates a connection to the remote end
};

/** \brief Connection endpoint.
 * This class represents a connection endpoint that the remote side of the connection can pass
 * messages to. Connections are handshaked by passing an instance of this type to the other
 * side while simultaneously receiving one from there.
 */
struct fusionsc_LvnEndPoint {
	uint16_t version = 1;
	
	void (*incRef)(fusionsc_LvnEndPoint*); // Increases the reference count of this object
	void (*decRef)(fusionsc_LvnEndPoint*); // Decreases the reference count of this object
	
	//! Notify other side that no more messages will be sent.
	void (*close)(fusionsc_LvnEndPoint*); 
	
	//! Send message across connection.
	void (*receive)(fusionsc_LvnEndPoint*, fusionsc_LvnMessage*);
};

//! Represents a segment owned by a Cap'n'proto message.
struct fusionsc_SegmentInfo {
	const void* data;
	size_t sizeInWords;
};

/**
 * \brief Represents an incoming Cap'n'proto messages with attached file descriptors.
 *
 * \warning The lifetime of the file descriptors is not tied to the lifetime of this message.
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

 /** @} */