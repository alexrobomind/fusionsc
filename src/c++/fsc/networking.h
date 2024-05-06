#pragma once

#include "common.h"

#include <kj/async-io.h>
#include <kj/compat/http.h>

#include <fsc/networking.capnp.h>

namespace kj {
	struct Network;
}

namespace fsc {

//! This class provides the high-level networking services (SSH, RPC etc.) based on primitives for forming simple network connections
struct NetworkInterfaceBase : public virtual NetworkInterface::Server {
	virtual Promise<Own<kj::AsyncIoStream>> makeConnection(kj::StringPtr host, unsigned int port) = 0;
	virtual Promise<Own<kj::ConnectionReceiver>> listen(kj::StringPtr host, Maybe<unsigned int> portHint = nullptr) = 0;
	
	Promise<void> sshConnect(SshConnectContext ctx) override;
	Promise<void> listen(ListenContext ctx) override;
	Promise<void> serve(ServeContext ctx) override;
	Promise<void> connect(ConnectContext ctx) override;
};

//! Network implementation based on the local network interface
struct LocalNetworkInterface : public NetworkInterfaceBase {
	//! Construct a network interface given a kj::Network (which can e.g. be used to restrict the address space)
	LocalNetworkInterface(Own<kj::Network> network);
	
	//! Construct network interface using the process-wide network
	LocalNetworkInterface();
	
	~LocalNetworkInterface();
	
	Promise<Own<kj::AsyncIoStream>> makeConnection(kj::StringPtr host, unsigned int port) override;
	Promise<Own<kj::ConnectionReceiver>> listen(kj::StringPtr host, Maybe<unsigned int> port) override;
	
	kj::Network& getNetwork();
	
private:
	Own<kj::Network> network;
};

NetworkInterface::OpenPort::Client listenViaHttp(Own<kj::ConnectionReceiver> receiver, NetworkInterface::Listener::Client target, Own<kj::HttpService> fallback);
NetworkInterface::OpenPort::Client listenViaHttp(Own<kj::ConnectionReceiver> receiver, capnp::Capability::Client target, Own<kj::HttpService> fallback);

}