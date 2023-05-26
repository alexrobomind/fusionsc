#include "ssh.h"
#include "local.h"

#include <libssh2.h>
#include <kj/list.h>

#include <errno.h>

using kj::AsyncIoStream;
using kj::List;

namespace fsc {

SSHChannel::~SSHChannel() {}
SSHChannelListener::~SSHChannelListener() {}
SSHSession::~SSHSession() {}
	
namespace {

struct LibSSH2 {
	LibSSH2() {
		libssh2_init(0); 
	}
	
	~LibSSH2() {
		libssh2_exit();
	}
	
	auto createSession(void* userData) {
		return libssh2_session_init_ex(nullptr, nullptr, nullptr, userData);
	}
};

static LibSSH2 LIBSSH2;
		
static ssize_t libssh2SendCallback(libssh2_socket_t sockfd, const void *buffer, size_t length, int flags, void **abstract);
static ssize_t libssh2RecvCallback(libssh2_socket_t sockfd, void *buffer, size_t length, int flags, void **abstract);

struct SSHChannelStream;
struct SSHChannelImpl;
struct SSHSessionImpl;

struct SSHChannelStream : public AsyncIoStream {
	SSHChannelStream(Own<SSHChannelImpl> channel, size_t streamID);
	~SSHChannelStream();
	
	// AsyncInputStream
	Promise<size_t> tryRead(void* buffer, size_t minBytes, size_t maxBytes) override;
	
	// AsyncOutputStream
	Promise<void> write(const void* buffer, size_t nBytes) override;
	Promise<void> write(ArrayPtr<const ArrayPtr<const byte>> pieces) override;
	
	// AsyncIoStream
	void shutdownWrite() override;
	Promise<void> whenWriteDisconnected() override;
	
	kj::ListLink<SSHChannelStream> streamsLink;
	kj::ListLink<SSHChannelStream> writingStreamsLink;
	Own<SSHChannelImpl> parent;
	size_t streamId;
	
	kj::Canceler canceler;
};

struct SSHChannelImpl : public SSHChannel, public kj::Refcounted {
	SSHChannelImpl(Own<SSHSessionImpl> parent, LIBSSH2_CHANNEL* channel);
	~SSHChannelImpl() noexcept;
	
	Own<SSHChannel> addRef() override { return kj::addRef(*this); };
	Own<AsyncIoStream> openStream(size_t id) override ;
	void close() override ;
	bool isOpen() override ;
	
	//! List of all open streams
	List<SSHChannelStream, &SSHChannelStream::streamsLink> streams;
	List<SSHChannelStream, &SSHChannelStream::writingStreamsLink> writingStreams;
		
	kj::ListLink<SSHChannelImpl> channelsLink;
	
	ForkedPromise<void> whenChannelWriteDisconnected;
	Own<PromiseFulfiller<void>> channelWriteDisconnectFulfiller;
	
	Own<SSHSessionImpl> parent;
	LIBSSH2_CHANNEL* const channel;
};

struct SSHChannelListenerImpl : public SSHChannelListener, public kj::Refcounted {
	SSHChannelListenerImpl(Own<SSHSessionImpl> parent, LIBSSH2_LISTENER* pListener, int port);
	~SSHChannelListenerImpl() noexcept;
	
	Promise<Own<SSHChannel>> accept() override ;
	int getPort() override { return port; }
	Own<SSHChannelListener> addRef() override { return kj::addRef(*this); };
	void close() override ;
	bool isOpen() override ;
	
	Own<SSHSessionImpl> parent;
	
	kj::ListLink<SSHChannelListenerImpl> listenersLink;
	LIBSSH2_LISTENER* const listener;
	const int port;
};
	
struct SSHSessionImpl : public SSHSession, public kj::Refcounted {
	SSHSessionImpl(Own<AsyncIoStream> stream);
	~SSHSessionImpl() noexcept;
	
	//! Performs basic handshaking
	Promise<void> startup();
	
	//! Repeatedly sends a keep-alive packet
	Promise<void> keepAlive();
	
	//! Attempts to read data from the underlying stream
	Maybe<size_t> tryReadFromStream(void* buffer, size_t length);
	
	//! Attempts to write data to the underlying stream
	size_t writeToStream(const void* buffer, size_t length);
	
	//! Check all ops for completion
	void checkAllOps();

	//! Helper class that keeps track of unfinished operations
	struct QueuedOp {
		kj::ListLink<QueuedOp> opsLink;
		
		virtual bool check() = 0;
		virtual void kill() = 0;
		virtual ~QueuedOp() {};
	};
	kj::List<QueuedOp, &QueuedOp::opsLink> queuedOps;
	
	template<typename T>
	Promise<T> queueOp(kj::Function<Maybe<T>()> op, bool cancelable = true);
	
	Promise<void> queueLibssh2Call(kj::Function<int()> op, bool cancelable = true);
	
	// --- Session interface ---
	
	Own<SSHSession> addRef() override { return kj::addRef(*this); }
	Promise<Own<SSHChannel>> connectRemote(kj::StringPtr remoteHost, size_t remotePort) override ;
	Promise<Own<SSHChannel>> connectRemote(kj::StringPtr remoteHost, size_t remotePort, kj::StringPtr srcHost, size_t srcPort) override ;
	Promise<Own<SSHChannelListener>> listen(kj::StringPtr host = "0.0.0.0"_kj, Maybe<int> port = nullptr) override ;
	Promise<bool> authenticatePassword(kj::StringPtr user, kj::StringPtr password) override;
	Promise<bool> authenticatePubkeyFile(kj::StringPtr user, kj::StringPtr pubkeyFile, kj::StringPtr privkeyFile, kj::StringPtr passPhrase) override;
	Promise<bool> authenticatePubkeyData(kj::StringPtr user, kj::StringPtr pubkeyData, kj::StringPtr privkeyData, kj::StringPtr passPhrase) override;
	bool isAuthenticated() override;
	void close() override ;
	bool isOpen() override ;
	Promise<void> drain() override ;
	
	//! Underlying I/O stream
	Own<AsyncIoStream> stream;
	LIBSSH2_SESSION* session;
	
	//! Promise that resolves when all writes are performed
	Promise<void> writesFinished = READY_NOW;
	
	//! Promise that tracks the currently active read
	Promise<void> activeRead = READY_NOW;
	
	//! Promise holding the keep-alive loop
	Promise<void> keepAliveTask;
	
	Maybe<size_t> bytesReady;
	size_t bytesConsumed = 0;
	kj::Array<kj::byte> readBuffer;
	bool readActive = false;
	
	List<SSHChannelImpl,&SSHChannelImpl::channelsLink> channels;
	List<SSHChannelListenerImpl, &SSHChannelListenerImpl::listenersLink> listeners;

	kj::TaskSet tasks;
};

// === Functions ===

ssize_t libssh2SendCallback(
	libssh2_socket_t sockfd,
	const void *buffer,
	size_t length,
	int flags,
	void **abstract
) {
	SSHSessionImpl* session = reinterpret_cast<SSHSessionImpl*>(*abstract);
	
	if(session -> stream.get() == nullptr) {
		return LIBSSH2_ERROR_SOCKET_DISCONNECT;
	}
	
	try {
		return (ssize_t) session -> writeToStream(buffer, length);
	} catch(kj::Exception& e) {
		KJ_DBG("Error in libssh2SendCallback", e);
		if(e.getType() == kj::Exception::Type::DISCONNECTED) {
			return LIBSSH2_ERROR_SOCKET_DISCONNECT;
		}
	}
	
	return LIBSSH2_ERROR_SOCKET_NONE;
}

ssize_t libssh2RecvCallback(
	libssh2_socket_t sockfd,
	void *buffer,
	size_t length,
	int flags,
	void **abstract
) {
	SSHSessionImpl* session = reinterpret_cast<SSHSessionImpl*>(*abstract);
	
	if(session -> stream.get() == nullptr) {
		return LIBSSH2_ERROR_SOCKET_DISCONNECT;
	}
	
	try {
		KJ_IF_MAYBE(pSize, session -> tryReadFromStream(buffer, length)) {
			return (ssize_t) *pSize;
		} else {
			return -EAGAIN;
		}
	} catch(kj::Exception& e) {
		KJ_DBG("Error in libssh2RecvCallback", e);
		if(e.getType() == kj::Exception::Type::DISCONNECTED) {
			return LIBSSH2_ERROR_SOCKET_DISCONNECT;
		}
	}
	
	return LIBSSH2_ERROR_SOCKET_NONE;
}
	

// === class SSHSessionIpml ===

namespace {
	struct NullErrorHandler : public kj::TaskSet::ErrorHandler {
		void taskFailed(kj::Exception&& exception) override {};
		
		static NullErrorHandler INSTANCE;
	};
	
	NullErrorHandler NullErrorHandler::INSTANCE;
}

SSHSessionImpl::SSHSessionImpl(Own<kj::AsyncIoStream> stream) :
	stream(mv(stream)),
	session(LIBSSH2.createSession(reinterpret_cast<void*>(this))),
	tasks(NullErrorHandler::INSTANCE),
	keepAliveTask(READY_NOW)
{
	// Set send and receive callbacks
	libssh2_session_callback_set(session, LIBSSH2_CALLBACK_SEND, (void*) &libssh2SendCallback);
	libssh2_session_callback_set(session, LIBSSH2_CALLBACK_RECV, (void*) &libssh2RecvCallback);
	
	// Set session to non-blocking operation
	libssh2_session_set_blocking(session, 0);
	
	libssh2_trace(session, LIBSSH2_TRACE_ERROR | LIBSSH2_TRACE_PUBLICKEY);
}

SSHSessionImpl::~SSHSessionImpl() noexcept {
	KJ_DBG("Closing session");
	// Terminate the keep-alive loop
	keepAliveTask = READY_NOW;
	
	// Remove the stream to cause all send / receive calls to fail
	// Note that this is safe because libssh2SendCallback and libssh2RecvCallback
	// explicitly check whether the stream is null (don't want an exception to be
	// propagated in that case).
	writesFinished = READY_NOW;
	stream = Own<AsyncIoStream>();
	
	// Close session
	// This also unlinks all nested objects
	close();
	
	// Terminate all active closing tasks (data structures will be cleared
	// up by libssh2_session_free
	tasks.clear();
	
	if(session != nullptr) {
		libssh2_session_free(session);
		session = nullptr;
	}
	KJ_DBG("Session closed");
}

Promise<void> SSHSessionImpl::startup() {
	return queueLibssh2Call([this]() {
		auto rc = libssh2_session_startup(session, 0);
		return rc;
	});
}

Promise<void> SSHSessionImpl::keepAlive() {
	return queueOp<int>([this]() -> Maybe<int> {
		int secondsToNext = 0;
		auto result = libssh2_keepalive_send(session, &secondsToNext);
		
		if(result != 0) {
			KJ_REQUIRE(result == LIBSSH2_ERROR_EAGAIN, "Failed to send keepalive packet");
			return nullptr;
		}
		
		return secondsToNext;
	})
	.then([](int seconds) {
		return getActiveThread().timer().afterDelay(seconds * kj::SECONDS);
	})
	.then([this]() {
		return keepAlive();
	});
}

Maybe<size_t> SSHSessionImpl::tryReadFromStream(void* buffer, size_t length) {
	KJ_IF_MAYBE(pBytes, bytesReady) {
		KJ_REQUIRE(!readActive);
		
		size_t bytesToCopy = kj::min(*pBytes - bytesConsumed, length);
		memcpy(buffer, readBuffer.begin() + bytesConsumed, bytesToCopy);
		// KJ_DBG("Read: ", readBuffer.slice(bytesConsumed, bytesToCopy));
		bytesConsumed += bytesToCopy;
		
		if(bytesConsumed >= *pBytes) {
			bytesConsumed = 0;
			bytesReady = nullptr;
		}
		
		return bytesToCopy;
	}
	
	if(!readActive) {
		readActive = true;
		
		activeRead = activeRead
		.then([this, length]() { 
			if(readBuffer.size() < length)
				readBuffer = kj::heapArray<kj::byte>(length);
			return stream->read(readBuffer.begin(), 1, length);
		})
		.then([this](size_t actualBytesRead) {
			// KJ_DBG("Read from stream: ", actualBytesRead);
			
			readActive = false;
			bytesReady = actualBytesRead;
			bytesConsumed = 0;
			checkAllOps();
		}).eagerlyEvaluate(nullptr);
	}
	
	return nullptr;
}

size_t SSHSessionImpl::writeToStream(const void* buffer, size_t length) {	
	auto tmpBuf = kj::heapArray<kj::byte>(length);
	memcpy(tmpBuf.begin(), buffer, length);
	
	writesFinished = writesFinished.then([this, tmpBuf = mv(tmpBuf)]() {
		// KJ_DBG("Writing data: ", tmpBuf);
		return stream->write(tmpBuf.begin(), tmpBuf.size());
	}).eagerlyEvaluate(nullptr);
	return length;
}

void SSHSessionImpl::checkAllOps() {
	for(QueuedOp& op : queuedOps) {
		if(session == nullptr) {
			op.kill();
			continue;
		}
		
		if(op.check()) {
			op.kill();
		}
	}
}

template<typename T>
Promise<T> SSHSessionImpl::queueOp(kj::Function<Maybe<T>()> op, bool cancelable) {
	struct Adapter : public QueuedOp {
		kj::Function<Maybe<T>()> op;
		kj::PromiseFulfiller<T>& fulfiller;
		SSHSessionImpl& session;
		
		bool cancelable;
		
		void unlink() {
			if(opsLink.isLinked()) {
				fulfiller.reject(KJ_EXCEPTION(FAILED, "Operation deleted"));
				session.queuedOps.remove(*this);
			}
		}
		
		bool check() override {
			bool finished = false;
			
			if(cancelable && !fulfiller.isWaiting())
				return true;
			
			bool ok = fulfiller.rejectIfThrows([this, &finished]() {
				KJ_IF_MAYBE(pResult, op()) {
					fulfiller.fulfill(mv(*pResult));
					finished = true;
				}
			});
			
			if(!ok) {
				finished = true;
			}
						
			return finished;
		}
		
		void kill() override {
			unlink();
			fulfiller.reject(KJ_EXCEPTION(FAILED, "Session closed"));
		}
		
		Adapter(PromiseFulfiller<T>& fulfiller, kj::Function<Maybe<T>()> op, SSHSessionImpl& session, bool cancelable) :
			op(mv(op)),
			fulfiller(fulfiller),
			session(session),
			cancelable(cancelable)
		{
			session.queuedOps.add(*this);
		}
		
		~Adapter() noexcept {
			unlink();
		}
	};
	
	auto result = kj::newAdaptedPromise<T, Adapter>(mv(op), *this, cancelable);
	checkAllOps();
	return result;
}

Promise<void> SSHSessionImpl::queueLibssh2Call(kj::Function<int()> op, bool uncancelable) {
	auto wrapped = [op = mv(op)]() mutable -> Maybe<int> {
		int callResult = op();
		
		if(callResult == LIBSSH2_ERROR_EAGAIN)
			return nullptr;
		
		return callResult;
	};
	
	return queueOp<int>(mv(wrapped), uncancelable)
	.then([](int result) {
		KJ_REQUIRE(result == 0, "Error in LibSSH2 call");
	});
}

Promise<bool> SSHSessionImpl::authenticatePassword(kj::StringPtr user, kj::StringPtr password) {
	return queueOp<bool>([this, user = kj::heapString(user), pw = kj::heapString(password)]() -> Maybe<bool> {
		auto rc = libssh2_userauth_password_ex(session, user.cStr(), user.size(), pw.cStr(), pw.size(), nullptr);
		
		if(rc == 0)
			return true;
		if(rc == LIBSSH2_ERROR_AUTHENTICATION_FAILED )
			return false;
		if(rc == LIBSSH2_ERROR_PASSWORD_EXPIRED ) {
			KJ_LOG(WARNING, "Authentication failed due to expired password");
			return false;
		}
		
		KJ_REQUIRE(rc == LIBSSH2_ERROR_EAGAIN, "Error during authentication");
		return nullptr;
	});
}

Promise<bool> SSHSessionImpl::authenticatePubkeyFile(kj::StringPtr user, kj::StringPtr pubkeyFile, kj::StringPtr privkeyFile, kj::StringPtr passPhrase) {
	auto fs = kj::newDiskFilesystem();
	auto curPath = fs -> getCurrentPath();
	
	auto pubPath = curPath.evalNative(pubkeyFile).toNativeString();
	auto privPath = curPath.evalNative(privkeyFile).toNativeString();
	
	return queueOp<bool>([
		this,
		user = kj::heapString(user),
		pubPath = kj::heapString(pubPath),
		privPath = kj::heapString(privPath),
		passPhrase = kj::heapString(passPhrase)
	]() -> Maybe<bool> {
		auto rc = libssh2_userauth_publickey_fromfile_ex(session, user.cStr(), user.size(), pubPath.cStr(), privPath.cStr(), passPhrase.cStr());
		
		if(rc == 0)
			return true;
		if(rc == LIBSSH2_ERROR_AUTHENTICATION_FAILED )
			return false;
		
		KJ_REQUIRE(rc == LIBSSH2_ERROR_EAGAIN, "Error during authentication");
		return nullptr;
	});
}

Promise<bool> SSHSessionImpl::authenticatePubkeyData(kj::StringPtr user, kj::StringPtr pubkeyData, kj::StringPtr privkeyData, kj::StringPtr passPhrase) {
	return queueOp<bool>([
		this,
		user = kj::heapString(user),
		pubData = kj::heapString(pubkeyData),
		privData= kj::heapString(privkeyData),
		passPhrase = kj::heapString(passPhrase)
	]() -> Maybe<bool> {
		auto rc = libssh2_userauth_publickey_frommemory(
			session,
			user.cStr(), user.size(),
			pubData.cStr(), pubData.size(),
			privData.cStr(), privData.size(),
			passPhrase.cStr()
		);
		
		KJ_DBG(pubData, privData, passPhrase);
		
		if(rc == 0)
			return true;
		if(rc == LIBSSH2_ERROR_AUTHENTICATION_FAILED) {
			char* errMsg;
			libssh2_session_last_error(session, &errMsg, nullptr, false);
			KJ_DBG(errMsg);
			return false;
		}
		
		if(rc != LIBSSH2_ERROR_EAGAIN) {
			char* errMsg;
			libssh2_session_last_error(session, &errMsg, nullptr, false);
			KJ_DBG(errMsg);
			return false;
		}
		
		KJ_REQUIRE(rc == LIBSSH2_ERROR_EAGAIN, "Error during authentication");
		return nullptr;
	});
	
}

bool SSHSessionImpl::isAuthenticated() {
	return libssh2_userauth_authenticated(session) != 0;
}

// --- Session interface ---

Promise<Own<SSHChannel>> SSHSessionImpl::connectRemote(kj::StringPtr remoteHost, size_t remotePort) {
	return queueOp<Own<SSHChannel>>([this, remoteHost = kj::heapString(remoteHost), remotePort]() mutable -> Maybe<Own<SSHChannel>> {
		LIBSSH2_CHANNEL* pChannel = libssh2_channel_direct_tcpip(session, remoteHost.cStr(), remotePort);
		
		if(pChannel == nullptr) {
			auto errCode = libssh2_session_last_errno(session);
			KJ_REQUIRE(errCode == LIBSSH2_ERROR_EAGAIN, "libssh2_channel_direct_tcpip failed", remoteHost, remotePort);
			
			return nullptr;			
		}
		
		return kj::refcounted<SSHChannelImpl>(kj::addRef(*this), pChannel);
	});
};

Promise<Own<SSHChannel>> SSHSessionImpl::connectRemote(kj::StringPtr remoteHost, size_t remotePort, kj::StringPtr srcHost, size_t srcPort) {
	return queueOp<Own<SSHChannel>>([this, remoteHost = kj::heapString(remoteHost), srcHost = kj::heapString(srcHost), srcPort, remotePort]() mutable -> Maybe<Own<SSHChannel>> {
		LIBSSH2_CHANNEL* pChannel = libssh2_channel_direct_tcpip_ex(session, remoteHost.cStr(), remotePort, srcHost.cStr(), srcPort);
		
		if(pChannel == nullptr) {
			auto errCode = libssh2_session_last_errno(session);
			KJ_REQUIRE(errCode == LIBSSH2_ERROR_EAGAIN, "libssh2_channel_direct_tcpip_ex failed", remoteHost, remotePort, srcHost, srcPort);
			
			return nullptr;			
		}
		
		return kj::refcounted<SSHChannelImpl>(kj::addRef(*this), pChannel);
	});
};

Promise<Own<SSHChannelListener>> SSHSessionImpl::listen(kj::StringPtr host, Maybe<int> port) {
	return queueOp<Own<SSHChannelListener>>([this, host = kj::heapString(host), port]() mutable -> Maybe<Own<SSHChannelListener>> {
		int boundPort = 0;
		
		int portArg = 0;
		KJ_IF_MAYBE(pPort, port) {
			portArg = *pPort;
		}
		
		// LIBSSH2_LISTENER * libssh2_channel_forward_listen_ex(LIBSSH2_SESSION *session, char *host, int port, int *bound_port, int queue_maxsize); 
		LIBSSH2_LISTENER* pListener =  libssh2_channel_forward_listen_ex(session, host.cStr(), portArg, &boundPort, 5);
		
		if(pListener == nullptr) {
			auto errCode = libssh2_session_last_errno(session);
			KJ_REQUIRE(errCode == LIBSSH2_ERROR_EAGAIN, "libssh2_channel_forward_listen_ex failed", host, portArg);
			
			return nullptr;			
		}
		
		return kj::refcounted<SSHChannelListenerImpl>(kj::addRef(*this), pListener, boundPort);
	});
};

void SSHSessionImpl::close() {
	if(session == nullptr)
		return;
	
	kj::Vector<Promise<void>> channelCloseOps;
	
	for(auto& listener : listeners) {
		listener.close();
	}
	
	for(auto& channel : channels) {
		channel.close();
	}
	
	tasks.add(
		queueLibssh2Call([this]() mutable {
			return libssh2_session_disconnect(session, "Disconnected by client");
		}, true)
		.then([this]() mutable {
			return queueLibssh2Call([this]() mutable {
				int result = libssh2_session_free(session);
				if(result == 0) session = nullptr;
				return result;
			}, true);
		})
	);
}

bool SSHSessionImpl::isOpen() {
	return session != nullptr;
}

Promise<void> SSHSessionImpl::drain() {
	return tasks.onEmpty();
}

// class SSHChannelListenerImpl

SSHChannelListenerImpl::SSHChannelListenerImpl(Own<SSHSessionImpl> newParent, LIBSSH2_LISTENER* listener, int port) :
	parent(mv(newParent)),
	listener(listener),
	port(port)
{
	parent -> listeners.add(*this);
}

SSHChannelListenerImpl::~SSHChannelListenerImpl() noexcept {
	close();
}

Promise<Own<SSHChannel>> SSHChannelListenerImpl::accept() {	
	return parent -> queueOp<Own<SSHChannel>>([this]() mutable -> Maybe<Own<SSHChannel>> {
		KJ_REQUIRE(isOpen(), "Listener was closed");
		LIBSSH2_CHANNEL* result = libssh2_channel_forward_accept(listener);
		
		if(result == nullptr) {
			auto errCode = libssh2_session_last_errno(parent -> session);
			KJ_REQUIRE(errCode == LIBSSH2_ERROR_EAGAIN, "libssh2_channel_forward_accept", port);
			
			return nullptr;			
		}
		
		return kj::refcounted<SSHChannelImpl>(kj::addRef(*parent), result);
	}).attach(addRef());
}

void SSHChannelListenerImpl::close() {
	if(listenersLink.isLinked()) {
		parent -> listeners.remove(*this);
		parent -> tasks.add(parent -> queueLibssh2Call([listener = listener]() {
			 return libssh2_channel_forward_cancel(listener);
		}));
	}
}

bool SSHChannelListenerImpl::isOpen() {
	return listenersLink.isLinked();
}

// class SSHChannelImpl

SSHChannelImpl::SSHChannelImpl(Own<SSHSessionImpl> newParent, LIBSSH2_CHANNEL* channel) :
	parent(mv(newParent)),
	channel(channel),
	whenChannelWriteDisconnected(nullptr)
{
	auto paf = kj::newPromiseAndFulfiller<void>();
	whenChannelWriteDisconnected = paf.promise.fork();
	channelWriteDisconnectFulfiller = mv(paf.fulfiller);
	
	parent -> channels.add(*this);
}

SSHChannelImpl::~SSHChannelImpl() noexcept {
	try {
		close();
	} catch(kj::Exception e) {
		KJ_DBG("Caught exception in SSHChannel destructor", e);
	}
}

void SSHChannelImpl::close() {
	if(!isOpen())
		return;
	
	channelWriteDisconnectFulfiller -> fulfill();
	
	parent -> tasks.add(parent -> queueLibssh2Call([channel = channel]() {
		return libssh2_channel_close(channel);
	}).attach(addRef()));
}

bool SSHChannelImpl::isOpen() {
	return channelsLink.isLinked();
}

Own<AsyncIoStream> SSHChannelImpl::openStream(size_t id) {
	return kj::heap<SSHChannelStream>(kj::addRef(*this), id);
}

// class SSHChannelStream

SSHChannelStream::SSHChannelStream(Own<SSHChannelImpl> newParent, size_t id) :
	parent(mv(newParent)),
	streamId(id)
{
	parent -> streams.add(*this);
	parent -> writingStreams.add(*this);
}

SSHChannelStream::~SSHChannelStream() {
	shutdownWrite();
	
	if(streamsLink.isLinked()) {
		parent -> streams.remove(*this);
	}
}

Promise<void> SSHChannelStream::whenWriteDisconnected() {
	return parent -> whenChannelWriteDisconnected.addBranch();
}

Promise<size_t> SSHChannelStream::tryRead(void* buffer, size_t minBytes, size_t maxBytes) {	
	return parent -> parent -> queueOp<size_t>([this, buffer = (char*) buffer, minBytes, maxBytes, bytesRead = (size_t) 0]() mutable -> Maybe<size_t> {
		KJ_REQUIRE(streamsLink.isLinked(), "Stream closed");
		KJ_REQUIRE(parent -> isOpen(), "Channel closed");
		
		// Try to read
		ssize_t rc = libssh2_channel_read_ex(parent -> channel, streamId, buffer, maxBytes);
		
		// No packets in queue, please ask again later
		if(rc == LIBSSH2_ERROR_EAGAIN)
			return nullptr;
		
		// If the channel was already closed, return whatever bytes we have
		if(rc == LIBSSH2_ERROR_CHANNEL_CLOSED)
			return bytesRead;
				
		KJ_REQUIRE(rc >= 0, "Failure during read", streamId);
		bytesRead += rc;
		
		if(rc < minBytes) {
			// We read data, but not enough to fill our buffer
			// Adjust buffer and boundaries for later calls
			buffer += rc;
			minBytes -= rc;
			maxBytes -= rc;
			
			// The read was insufficient to meet our window requirements.
			// See whether that is because the stream found EOF
			auto rc = libssh2_channel_eof(parent -> channel);
			if(rc == 1) {
				// We hit EOF
				return bytesRead;
			} else if(rc == 0) {
				// The stream is still open
				// Wait until more data arrives
				return nullptr;
			} else if(rc == LIBSSH2_ERROR_EAGAIN) {
				// Stream is busy, please ask again later
				// Wait until data arrives
				return nullptr;
			}
			
			KJ_FAIL_REQUIRE("libssh2_channel_eof failed", rc, streamId);
		} else {
			// We collected enough bytes for now.
			return bytesRead;
		}
	});
}

Promise<void> SSHChannelStream::write(const void* buffer, size_t nBytes) {
	return parent -> parent -> queueOp<bool>([this, buffer = (const char*) buffer, nBytes]() mutable -> Maybe<bool> {
		KJ_REQUIRE(streamsLink.isLinked(), "Stream closed");
		KJ_REQUIRE(parent -> isOpen(), "Channel closed");
			
		ssize_t rc = libssh2_channel_write_ex(parent -> channel, streamId, buffer, nBytes);
		
		if(rc == LIBSSH2_ERROR_EAGAIN)
			return nullptr;
		
		KJ_REQUIRE(rc >= 0);
		buffer += rc;
		nBytes -= rc;
		
		if(nBytes == 0)
			return true;
		
		return nullptr;
	}).ignoreResult();
}

Promise<void> SSHChannelStream::write(ArrayPtr<const ArrayPtr<const byte>> pieces) {	
	Promise<void> result = READY_NOW;
	
	for(auto piece : pieces) {
		result = result.then([this, piece]() {
			write(piece.begin(), piece.size());
		});
	}
	
	return result;
}

void SSHChannelStream::shutdownWrite() {
	if(!writingStreamsLink.isLinked())
		return;
	
	parent -> writingStreams.remove(*this);
	
	if(parent -> writingStreams.empty()) {
		parent -> parent -> tasks.add(parent -> parent -> queueLibssh2Call([this]() mutable {
			KJ_REQUIRE(streamsLink.isLinked(), "Stream closed");
			KJ_REQUIRE(parent -> isOpen(), "Channel closed");
			
			return libssh2_channel_send_eof(parent -> channel);
		}));
		parent -> channelWriteDisconnectFulfiller -> fulfill();
	}
}

}

Promise<Own<SSHSession>> createSSHSession(Own<kj::AsyncIoStream> stream) {
	Own<SSHSessionImpl> sess = kj::refcounted<SSHSessionImpl>(mv(stream));
	Promise<void> startupTask = sess -> startup();
	return startupTask
	.then([sess = mv(sess)]() mutable -> Own<SSHSession> { return mv(sess); });
}

}