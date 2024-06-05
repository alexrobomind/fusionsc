#include "ssh.h"
#include "networking.h"
#include "local.h"
#include "data.h"
#include "services.h"

#include <kj/compat/http.h>
#include <kj/compat/url.h>
// #include <kj/async-io.h>

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
using kj::HttpServer;
using kj::HttpClientSettings;
using kj::HttpServerSettings;
using kj::HttpService;
using kj::HttpClient;

using kj::AsyncIoStream;
using kj::ConnectionReceiver;

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
	
	Promise<void> close(CloseContext ctx) override {
		session -> close();
		return session -> drain();
	}
	
	Promise<void> authenticatePassword(AuthenticatePasswordContext ctx) override {
		auto params = ctx.getParams();
		return session -> authenticatePassword(params.getUser(), params.getPassword())
		.then([](bool result) {
			KJ_REQUIRE(result, "Authentication failed");
		});
	}
	
	Promise<void> authenticateKeyFile(AuthenticateKeyFileContext ctx) override {		
		auto params = ctx.getParams();
		return session -> authenticatePubkeyFile(
			params.getUser(), params.getPubKeyFile(),
			params.getPrivKeyFile(), params.getKeyPass()
		)
		.then([](bool result) {
			KJ_REQUIRE(result, "Authentication failed");
		});
	}
	
	// The following function will be enabled once https://github.com/libssh2/libssh2/issues/1047 is resolved.
	/*Promise<void> authenticateKeyData(AuthenticateKeyDataContext ctx) override {
		auto params = ctx.getParams();
		return session -> authenticatePubkeyData(
			params.getUser(), params.getPubKey(),
			params.getPrivKey(), params.getKeyPass()
		)
		.then([](bool result) {
			KJ_REQUIRE(result, "Authentication failed");
		});
	}*/
};

struct ConnectionMembrane : public MembranePolicy, Refcounted {
	Own<void> target;
	ForkedPromise<void> rev;
	
	ConnectionMembrane(Own<void> target, Promise<void> revIn) : target(mv(target)), rev(revIn.fork()) {}
	
	~ConnectionMembrane() {
		// If a connection is held open by the remote end, then the last remaining
		// references to this membrane will be dropped inside of the messageLoop() of
		// the RPC system. Immediately deleting the connection objects will cause
		// undefined behavior.
		// Instead, schedule a safe deletion inside a detached promise.
		getActiveThread().detach(kj::evalLater([target = mv(target)]() mutable {
			target = Own<void>();
		}));
	}
	
	// MembranePolicy interface
	
	Own<MembranePolicy> addRef() override { return kj::addRef(*this); }
	
	Maybe<Capability::Client> outboundCall(uint64_t interfaceId, uint16_t methodId, Capability::Client target) override {
		for(auto blockedId : protectedInterfaces()) {
			KJ_REQUIRE(interfaceId != blockedId, "Direct remote calls to protected interfaces are prohibited");
		}
		
		return nullptr;
	}
	
	Maybe<Capability::Client> inboundCall(uint64_t interfaceId, uint16_t methodId, Capability::Client target) override {		
		return nullptr;
	}
	
	Maybe<Promise<void>> onRevoked() override {	return rev.addBranch(); }
	
};

struct StreamNetworkConnection : public capnp::BootstrapFactory<capnp::rpc::twoparty::VatId>, Refcounted, public NetworkInterface::Connection::Server {
	using VatId = capnp::rpc::twoparty::VatId;
	using Side = capnp::rpc::twoparty::Side;
	
	// Constructors
	
	StreamNetworkConnection(Own<AsyncIoStream> newStream, kj::Function<Capability::Client()> factory, Side local, Side remote) :
		stream(mv(newStream)),
		vatNetwork(kj::heap<capnp::TwoPartyVatNetwork>(*(stream.get<Own<AsyncIoStream>>()), local)),
		factory(mv(factory)),
		peerSide(remote)
	{
		constructCommon();
	}
	
	StreamNetworkConnection(Own<MessageStream> newStream, kj::Function<Capability::Client()> factory, Side local, Side remote) :
		stream(mv(newStream)),
		vatNetwork(kj::heap<capnp::TwoPartyVatNetwork>(*(stream.get<Own<MessageStream>>()), local)),
		factory(mv(factory)),
		peerSide(remote)
	{
		constructCommon();
	}
	
	~StreamNetworkConnection() {
	}
	
	Own<MembranePolicy> membranePolicy() {
		return kj::refcounted<ConnectionMembrane>(kj::addRef(*this), canceler.wrap(Promise<void>(NEVER_DONE)));
	}
	
	void constructCommon() {
		rpcSystem.emplace(static_cast<capnp::TwoPartyVatNetworkBase&>(*vatNetwork), *this);
		
		disconnectHandler = vatNetwork -> onDisconnect()
		.then([this]() {
			disconnect(KJ_EXCEPTION(DISCONNECTED, "Connection closed remotely"));
			stream = nullptr;
		}).eagerlyEvaluate(nullptr);
	}
	
	void disconnect(kj::Exception e) {
		canceler.cancel(mv(e));
		rpcSystem = nullptr;
	}
	
	// Refcounting & remote interface access
		
	Capability::Client bootstrap(Side server) {
		Temporary<VatId> id;
		id.setSide(server);
		KJ_IF_MAYBE(pRpcSystem, rpcSystem) {
			return capnp::membrane(
				pRpcSystem -> bootstrap(id.asReader()),
				membranePolicy()
			);
		}
		return KJ_EXCEPTION(DISCONNECTED, "Connection already closed");
	}
	
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
		return capnp::reverseMembrane(factory(), membranePolicy());
	}
	
	OneOf<Own<AsyncIoStream>, Own<MessageStream>, std::nullptr_t> stream;
	Own<capnp::TwoPartyVatNetwork> vatNetwork;
	kj::Function<Capability::Client()> factory;
	Side peerSide;
	Maybe<capnp::RpcSystem<VatId>> rpcSystem;
	
	Promise<void> disconnectHandler = nullptr;
	
	kj::Canceler canceler;
};

struct OpenPortImpl : public NetworkInterface::OpenPort::Server {
	Maybe<Own<ConnectionReceiver>> receiver;
	Maybe<Own<HttpServer>> server;
	Promise<void> listenPromise;
	
	OpenPortImpl(Own<HttpServer> server, Own<ConnectionReceiver> recv) :
		receiver(mv(recv)), server(mv(server)), listenPromise(nullptr)
	{
		FSC_ASSERT_MAYBE(pSrv, this -> server, "Internal error");
		FSC_ASSERT_MAYBE(pRecv, this -> receiver, "Internal error");
		
		listenPromise = (*pSrv) -> listenHttp(**pRecv).eagerlyEvaluate(nullptr);
	}
	
	Promise<void> getInfo(GetInfoContext ctx) override {
		FSC_REQUIRE_MAYBE(pRecv, receiver, "Listener already stopped");
		
		ctx.getResults().setPort((*pRecv) -> getPort());
		return READY_NOW;
	}
	
	Promise<void> drain(DrainContext ctx) override {
		FSC_MAYBE_OR_RETURN(pSrv, server, READY_NOW);
		
		return (*pSrv) -> drain();
	}
	
	Promise<void> stopListening(StopListeningContext ctx) override {
		listenPromise = READY_NOW;
		receiver = nullptr;
		
		return READY_NOW;
	}
	
	Promise<void> closeAll(CloseAllContext) override {
		KJ_UNIMPLEMENTED("Safe shutdown not yet implemented");
	}
	
	Promise<void> unsafeCloseAllNow(UnsafeCloseAllNowContext) override {
		listenPromise = READY_NOW;
		server = nullptr;
		receiver = nullptr;
		return READY_NOW;
	}
};

static kj::HttpHeaderTable DEFAULT_HEADERS = kj::HttpHeaderTable();

struct SimpleMessageFallback : public SimpleHttpServer::Server {
	Promise<void> serve(ServeContext ctx) {
		unsigned int UPGRADE_REQUIRED = 426;
		
		auto results = ctx.initResults();
		results.setStatus(UPGRADE_REQUIRED);
		results.setStatusText("Upgrade required");
		results.setBody(
			"This is a FusionSC connection endpoint, which expects WebSocket requests."
			" Please connect to it using the FusionSC client."
			" See: https://jugit.fz-juelich.de/a.knieps/fsc for more information"
		);
		return READY_NOW;
	}
};

struct FallbackWrapper : public kj::HttpService {
	SimpleHttpServer::Client fallback;
	
	FallbackWrapper(SimpleHttpServer::Client clt) :
		fallback(mv(clt))
	{}
	
	Promise<void> request(
		HttpMethod method, kj::StringPtr url, const HttpHeaders& headers,
		kj::AsyncInputStream& requestBody, Response& response
	) {
		
		auto request = fallback.serveRequest();
		request.setMethod(kj::str(method));
		request.setUrl(url);
		
		return request.send().then([&response](auto simpleResponse) mutable {
			HttpHeaders responseHeaders(DEFAULT_HEADERS);
			responseHeaders.set(HttpHeaderId::UPGRADE, "websocket");
			responseHeaders.set(HttpHeaderId::CONTENT_TYPE, "text/html; charset=utf-8");
							
			auto outputStream = response.send(simpleResponse.getStatus(), simpleResponse.getStatusText(), responseHeaders, simpleResponse.getBody().size());
			auto sendPromise = outputStream -> write(simpleResponse.getBody().begin(), simpleResponse.getBody().size());
			return sendPromise.attach(mv(outputStream), mv(simpleResponse));
		});
	}
};

struct HttpListener : public kj::HttpService {
	using Side = capnp::rpc::twoparty::Side;
	
	OneOf<NetworkInterface::Listener::Client, capnp::Capability::Client> listener;
	Own<kj::HttpService> fallback;
	
	HttpListener(NetworkInterface::Listener::Client listener, Own<kj::HttpService> fallback) :
		listener(mv(listener)), fallback(mv(fallback))
	{}
	
	HttpListener(capnp::Capability::Client client, Own<kj::HttpService> fallback) :
		listener(mv(client)), fallback(mv(fallback))
	{}
	
	kj::Promise<void> request(
		HttpMethod method,
		kj::StringPtr url,
		const HttpHeaders& headers,
		kj::AsyncInputStream& requestBody,
		Response& response
	) override {
		HttpHeaders responseHeaders(DEFAULT_HEADERS);
		responseHeaders.set(HttpHeaderId::UPGRADE, "websocket");
		
		// Check if the request is a websocket request
		if(!headers.isWebSocket()) {
			return fallback -> request(method, url, headers, requestBody, response);
		}
		
		// Transition to a WebSocket response
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
		return nc -> vatNetwork -> onDisconnect().attach(kj::addRef(*nc));
	}
};

struct DefaultEntropySource : public kj::EntropySource  {
	static DefaultEntropySource INSTANCE;
	void generate(kj::ArrayPtr<byte> buffer) override {
		getActiveThread().rng().randomize(buffer);
	}
};

DefaultEntropySource DefaultEntropySource::INSTANCE;

NetworkInterface::Connection::Client connectViaHttp(Own<AsyncIoStream> stream, kj::StringPtr host, kj::StringPtr url, bool allowCompression) {
	HttpClientSettings settings;
	settings.entropySource = DefaultEntropySource::INSTANCE;
	settings.webSocketCompressionMode = allowCompression ? HttpClientSettings::AUTOMATIC_COMPRESSION : HttpClientSettings::NO_COMPRESSION;
	
	auto client = ownHeld(kj::newHttpClient(DEFAULT_HEADERS, *stream, settings));
	client.attach(mv(stream));
	
	kj::HttpHeaders headers(DEFAULT_HEADERS);
	headers.add("Host", host);
	
	return client -> openWebSocket(url, headers)
	.then([client = client.x()](HttpClient::WebSocketResponse response) mutable -> fsc::NetworkInterface::Connection::Client {
		KJ_REQUIRE(
			response.webSocketOrBody.is<Own<kj::WebSocket>>(),
			"Connection did not provide a proper websocket",
			response.statusCode, response.statusText
		);
		
		Own<kj::WebSocket> webSocket = mv(response.webSocketOrBody.get<Own<kj::WebSocket>>());
		auto msgStream = kj::heap<capnp::WebSocketMessageStream>(*webSocket);
		msgStream = msgStream.attach(mv(webSocket), mv(client));
		
		using capnp::rpc::twoparty::Side;
		return kj::refcounted<StreamNetworkConnection>(mv(msgStream), []() { return Capability::Client(nullptr); }, Side::CLIENT, Side::SERVER);
	});
}

}

// === class NetworkInterfaceBase ===

Promise<void> NetworkInterfaceBase::sshConnect(SshConnectContext ctx) {
	auto params = ctx.getParams();
	return makeConnection(params.getHost(), params.getPort())
	.then([](Own<kj::AsyncIoStream> stream) {
		return createSSHSession(mv(stream));
	})
	.then([ctx](Own<SSHSession> sshSession) mutable {
		ctx.initResults().setConnection(kj::heap<SSHConnectionImpl>(mv(sshSession)));
	});
}

Promise<void> NetworkInterfaceBase::listen(ListenContext ctx) {
	auto params = ctx.getParams();
	
	Maybe<unsigned int> portHint;
	if(params.getPortHint() != 0)
		portHint = params.getPortHint();
	
	SimpleHttpServer::Client fallback = params.hasFallback() ?
		params.getFallback() :
		kj::heap<SimpleMessageFallback>()
	;
	
	return listen(ctx.getParams().getHost(), portHint)
	.then([ctx, params, fallback](Own<ConnectionReceiver> recv) mutable {
		ctx.getResults().setOpenPort(listenViaHttp(mv(recv), params.getListener(), kj::heap<FallbackWrapper>(fallback)));
	});
}

Promise<void> NetworkInterfaceBase::serve(ServeContext ctx) {
	auto params = ctx.getParams();
	
	Maybe<unsigned int> portHint;
	if(params.getPortHint() != 0)
		portHint = params.getPortHint();
	
	SimpleHttpServer::Client fallback = params.hasFallback() ?
		params.getFallback() :
		kj::heap<SimpleMessageFallback>()
	;
	
	return listen(ctx.getParams().getHost(), portHint)
	.then([ctx, params, fallback](Own<ConnectionReceiver> recv) mutable {
		ctx.getResults().setOpenPort(listenViaHttp(mv(recv), params.getServer(), kj::heap<FallbackWrapper>(fallback)));
	});
}

Promise<void> NetworkInterfaceBase::connect(ConnectContext ctx) {
	using kj::Url;
	Url url = Url::parse(ctx.getParams().getUrl());
	
	// URL compatibility validation
	KJ_IF_MAYBE(pDontCare, url.userInfo) {
		KJ_FAIL_REQUIRE("User authentication via HTTP is hilariously unsafe and not supported");
	}
	
	KJ_REQUIRE(url.scheme == "http" || url.scheme == "ws", "Only url schemes 'http' and 'ws' are supported. HTTPS support is currently not available");
	
	auto hostAndPort = url.host.asPtr();
	
	int port = 80;
	kj::String host = nullptr;
	KJ_IF_MAYBE(pIdx, hostAndPort.findFirst(':')) {
		host = kj::heapString(hostAndPort.slice(0, *pIdx));
		port = hostAndPort.slice(*pIdx + 1).parseAs<unsigned int>();
	} else {
		host = kj::heapString(hostAndPort);
	}
	
	return makeConnection(host, port)
	.then([ctx, url = mv(url), host = kj::heapString(host)](Own<kj::AsyncIoStream> stream) mutable {
		kj::String httpUrl = url.toString(Url::HTTP_REQUEST);
		ctx.getResults().setConnection(connectViaHttp(mv(stream), host, httpUrl, ctx.getParams().getAllowCompression()));
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
	auto newHost = kj::heapString(host);
	auto result = network -> parseAddress(newHost, port)
	.then([](Own<kj::NetworkAddress> addr) {
		return addr -> connect();
	});
	return result.attach(mv(newHost));
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

NetworkInterface::OpenPort::Client listenViaHttp(Own<kj::ConnectionReceiver> receiver, NetworkInterface::Listener::Client target, Own<kj::HttpService> fallback) {
	// Wrap listener client into HTTP server
	HttpServerSettings settings;
	settings.webSocketCompressionMode = HttpServerSettings::AUTOMATIC_COMPRESSION;
	
	auto service = kj::heap<HttpListener>(mv(target), mv(fallback));
	auto server = kj::heap<kj::HttpServer>(getActiveThread().timer(), DEFAULT_HEADERS, *service, settings);
	server = server.attach(mv(service));
	
	// Create open port interface
	return kj::heap<OpenPortImpl>(mv(server), mv(receiver));
}

NetworkInterface::OpenPort::Client listenViaHttp(Own<kj::ConnectionReceiver> receiver, capnp::Capability::Client target, Own<kj::HttpService> fallback) {
	// Wrap listener client into HTTP server
	HttpServerSettings settings;
	settings.webSocketCompressionMode = HttpServerSettings::AUTOMATIC_COMPRESSION;
	
	// Wrap listener client into HTTP server
	auto service = kj::heap<HttpListener>(mv(target), mv(fallback));
	auto server = kj::heap<kj::HttpServer>(getActiveThread().timer(), DEFAULT_HEADERS, *service, settings);
	server = server.attach(mv(service));
	
	// Create open port interface
	return kj::heap<OpenPortImpl>(mv(server), mv(receiver));
}

}
