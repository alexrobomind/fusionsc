#include "ssh.h"
#include "networking.h"
#include "local.h"
#include "data.h"
#include "services.h"

#include <kj/compat/http.h>
#include <capnp/compat/websocket-rpc.h>
#include <capnp/membrane.h>
#include <capnp/rpc-twoparty.h>

using kj::Refcounted;
using kj::AsyncIoStream;

using capnp::MembranePolicy;
using capnp::MessageStream;
using capnp::Capability;

using kj::HttpMethod;
using kj::HttpHeaders;
using kj::HttpHeaderId;

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

struct StreamNetworkConnection : public MembranePolicy, capnp::BootstrapFactory<capnp::rpc::twoparty::VatId>, Refcounted, public NetworkInterface::Connection::Server {
	using VatId = capnp::rpc::twoparty::VatId;
	using Side = capnp::rpc::twoparty::Side;
	
	// Constructors
	
	StreamNetworkConnection(Own<AsyncIoStream> newStream, kj::Function<Capability::Client()> factory, Side local, Side remote) :
		stream(mv(newStream)),
		vatNetwork(*(stream.get<Own<AsyncIoStream>>()), local),
		factory(mv(factory)),
		peerSide(remote)
	{
		rpcSystem.emplace(static_cast<capnp::TwoPartyVatNetworkBase&>(vatNetwork), *this);
	}
	
	StreamNetworkConnection(Own<MessageStream> newStream, kj::Function<Capability::Client()> factory, Side local, Side remote) :
		stream(mv(newStream)),
		vatNetwork(*(stream.get<Own<MessageStream>>()), local),
		factory(mv(factory)),
		peerSide(remote)
	{
		rpcSystem.emplace(static_cast<capnp::TwoPartyVatNetworkBase&>(vatNetwork), *this);
	}
	
	void disconnect(kj::Exception e) {
		canceler.cancel(mv(e));
		rpcSystem = nullptr;
		stream = nullptr;
	}
	
	// Refcounting & remote interface access
		
	Capability::Client bootstrap(Side server) {
		Temporary<VatId> id;
		id.setSide(server);
		KJ_IF_MAYBE(pRpcSystem, rpcSystem) {
			return capnp::membrane(
				pRpcSystem -> bootstrap(id.asReader()),
				addRef()
			);
		}
		return KJ_EXCEPTION(DISCONNECTED, "Connection already closed");
	}
	
	// MembranePolicy interface
	
	Own<MembranePolicy> addRef() override { return addRef(); }
	
	Maybe<Capability::Client> outboundCall(uint64_t interfaceId, uint16_t methodId, Capability::Client target) override {
		for(auto blockedId : protectedInterfaces()) {
			KJ_REQUIRE(interfaceId != blockedId, "Direct remote calls to protected interfaces are prohibited");
		}
		
		return nullptr;
	}
	
	Maybe<Capability::Client> inboundCall(uint64_t interfaceId, uint16_t methodId, Capability::Client target) override {		
		return nullptr;
	}
	
	Maybe<Promise<void>> onRevoked() override {	return canceler.wrap<void>(NEVER_DONE); }
	
	// NetworkInterface::Connection interface
	
	Promise<void> getRemote(GetRemoteContext ctx) override {
		ctx.getResults().setRemote(bootstrap(peerSide));
		return READY_NOW;
	}
	
	Promise<void> close(CloseContext ctx) override {
		KJ_UNIMPLEMENTED("Safe closure not yet implemented");
	}

	Promise<void> unsafeCloseNow(UnsafeCloseNowContext ctx) override {
		disconnect(KJ_EXCEPTION(DISCONNECTED, "Connection closed"));
		return READY_NOW;
	}
	
	// BootstrapFactory interface
	
	Capability::Client createFor(typename VatId::Reader clientId) override {
		return capnp::reverseMembrane(factory(), addRef());
	}
	
	OneOf<Own<AsyncIoStream>, Own<MessageStream>, std::nullptr_t> stream;
	capnp::TwoPartyVatNetwork vatNetwork;
	kj::Function<Capability::Client()> factory;
	Side peerSide;
	Maybe<capnp::RpcSystem<VatId>> rpcSystem;
	
	kj::Canceler canceler;
};

struct FSCHttpService : public kj::HttpService {
	using Side = capnp::rpc::twoparty::Side;
	
	static inline kj::HttpHeaderTable headerTable = kj::HttpHeaderTable();
	
	OneOf<NetworkInterface::Listener::Client, capnp::Capability::Client> listener;
	
	FSCHttpService(NetworkInterface::Listener::Client listener) :
		listener(mv(listener))
	{}
	
	FSCHttpService(capnp::Capability::Client client) :
		listener(mv(client))
	{}
	
	kj::Promise<void> request(
		HttpMethod method,
		kj::StringPtr url,
		const HttpHeaders& headers,
		kj::AsyncInputStream& requestBody,
		Response& response
	) override {
		HttpHeaders responseHeaders(headerTable);
		responseHeaders.set(HttpHeaderId::UPGRADE, "websocket");
		
		// Check if the request is a websocket request
		if(!headers.isWebSocket()) {
			unsigned int UPGRADE_REQUIRED = 426;
			
			return response.sendError(
				UPGRADE_REQUIRED,
				"This is an FSC connection endpoint, which expects WebSocket requests."
				" Please connect to it using the fsc client."
				" See: https://jugit.fz-juelich.de/a.knieps/fsc for more information",
				responseHeaders
			);
		}
		
		// Transition to a WebSocket response
		responseHeaders.set(HttpHeaderId::CONNECTION, "Upgrade");
		auto wsStream = response.acceptWebSocket(responseHeaders);
		auto msgStream = kj::heap<capnp::WebSocketMessageStream>(*wsStream);
		msgStream = msgStream.attach(mv(wsStream));
		
		// Create network connection
		
		kj::Function<capnp::Capability::Client()> acceptFunc;
		
		if(listener.is<NetworkInterface::Listener::Client>()) {
			acceptFunc = [listener = cp(this->listener.get<NetworkInterface::Listener::Client>())]() mutable {
				return listener.acceptRequest().sendForPipeline().getClient();
			};
		} else {
			acceptFunc = [clt = cp(this->listener.get<capnp::Capability::Client>())]() mutable {
				return clt;
			};
		}
		
		auto nc = kj::refcounted<StreamNetworkConnection>(mv(msgStream), mv(acceptFunc), Side::SERVER, Side::CLIENT);
		
		KJ_IF_MAYBE(pRPC, nc -> rpcSystem) {
			return pRPC -> run().attach(mv(nc));
		}
		return READY_NOW;
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