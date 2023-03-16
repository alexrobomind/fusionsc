#include "ssh.h"
#include "networking.h"
#include "local.h"

namespace fsc {

namespace {

struct SSHForwardListener : public kj::ConnectionReceiver {
	Own<SSHChannelListener> backend;
	
	SSHForwardListener(Own<SSHChannelListener> backend) :
		backend(mv(backend))
	{}
	
	unsigned int getPort() override {
		return backend -> getPort();
	}
	
	Promise<Own<kj::AsyncIoStream>> accept() override {
		return backend -> accept()
		.then([](Own<SSHChannel> channel) {
			return channel -> openStream(0);
		});
	}
};

//! Network interface tunneled through an SSH connection
struct SSHConnectionImpl : public SSHConnection::Server, public NetworkInterfaceBase {
	Own<SSHSession> session;
	
	SSHConnectionImpl(Own<SSHSession> session) :
		session(mv(session))
	{}
	
	Promise<Own<kj::AsyncIoStream>> makeConnection(kj::StringPtr host, unsigned int port) override {
		KJ_REQUIRE(session -> isAuthenticated(), "Not authenticated");
		return session -> connectRemote(host, port)
		.then([](Own<SSHChannel> channel) {
			return channel -> openStream(0);
		});
	}
	
	Promise<Own<kj::ConnectionReceiver>> listen(kj::StringPtr host, Maybe<unsigned int> port) override {
		KJ_REQUIRE(session -> isAuthenticated(), "Not authenticated");
		return session -> listen(host, port)
		.then([](Own<SSHChannelListener> listener) -> Own<kj::ConnectionReceiver> {
			return kj::heap<SSHForwardListener>(mv(listener));
		});
	}
	
	Promise<void> close(CloseContext ctx) {
		session -> close();
		return session -> drain();
	}
	
	Promise<void> authenticatePassword(AuthenticatePasswordContext ctx) {
		auto params = ctx.getParams();
		return session -> authenticatePassword(params.getUser(), params.getPassword())
		.then([](bool result) {
			KJ_REQUIRE(result, "Authentication failed");
		});
	}
};

}

// === class NetworkInterfaceBase ===

Promise<void> NetworkInterfaceBase::sshConnect(SshConnectContext ctx) {
	auto params = ctx.getParams();
	return makeConnection(params.getHost(), params.getPort())
	.then([](Own<kj::AsyncIoStream> stream) {
		KJ_DBG("sshConnect: Connection formed");
		return createSSHSession(mv(stream));
	})
	.then([ctx](Own<SSHSession> sshSession) mutable {
		KJ_DBG("sshConnect: Session created");
		ctx.initResults().setConnection(kj::heap<SSHConnectionImpl>(mv(sshSession)));
	});
}

// === class LocalNetworkInterface ===

LocalNetworkInterface::LocalNetworkInterface() :
	LocalNetworkInterface(kj::attachRef(getActiveThread().network()))
{}

LocalNetworkInterface::LocalNetworkInterface(Own<kj::Network> network) :
	network(mv(network))
{}

LocalNetworkInterface::~LocalNetworkInterface() {}

Promise<Own<kj::AsyncIoStream>> LocalNetworkInterface::makeConnection(kj::StringPtr host, unsigned int port) {
	return network -> parseAddress(host, port)
	.then([](Own<kj::NetworkAddress> addr) {
		return addr -> connect();
	});
}

Promise<Own<kj::ConnectionReceiver>> LocalNetworkInterface::listen(kj::StringPtr host, Maybe<unsigned int> port) {
	unsigned int portArg = 0;
	KJ_IF_MAYBE(pPort, port) {
		portArg = *pPort;
	}
	
	return network -> parseAddress(host, portArg)
	.then([](Own<kj::NetworkAddress> addr) {
		return addr -> listen();
	});
}

kj::Network& LocalNetworkInterface::getNetwork() { return *network; }

}