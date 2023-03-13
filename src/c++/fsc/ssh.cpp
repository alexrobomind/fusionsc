#include "ssh.h"

#include <libssh2.h>

namespace fsc {
	
namespace {
		
static ssize_t libssh2SendCallback(libssh2_socket_t sockfd, const void *buffer, size_t length, int flags, void **abstract);
static ssize_t libssh2RecvCallback(libssh2_socket_t sockfd, const void *buffer, size_t length, int flags, void **abstract);

struct SSHChannelStream;
struct SSHChannelImpl;
struct SSHSessionImpl;

struct SSHChannelStream : public AsyncIOStream {
	SSHChannelStream(Own<SSHChannelImpl> channel, size_t streamID);
	~SSHChannelStream();
	
	// AsyncInputStream
	Promise<size_t> tryRead(void* buffer, size_t minBytes, size_t maxBytes) override;
	
	// AsyncOutputStream
	Promise<void> write(void* buffer, size_t nBytes) override;
	Promise<void> write(ArrayPtr<const ArrayPtr<const byte>> pieces) override;
	
	// AsyncIOStream
	void shutdownWrite() override;
	
	void close();
	
	kj::ListLink<SSHChannelStream> streamsLink;
	kj::ListLink<SSHChannelStream> writingStreamsLink;
	Own<SSHChannelImpl> parent;
	size_t streamId;
	
	kj::Canceler canceler;
};

struct SSHChannelImpl : public SSHChannel, public kj::Refcounted {
	SSHChannelImpl(Own<SSHSessionImpl> parent, LIBSSH2_CHANNEL* channel);
	~SSHChannelImpl();
	
	Own<SSHChannel> addRef() override { return kj::addRef(*this); };
	Own<AsyncIOStream> openStream(size_t id) override ;
	void close() override ;
	bool isOpen() override ;
	
	//! List of all open streams
	List<ChannelStream, &ChannelStream::streamsLink> streams;
	List<ChannelStream, &ChannelStream::writingStreamsLink> writingStreams;
		
	kj::ListLink<SSHChannelImpl> channelsLink;
	
	LIBSSH2_CHANNEL* const libChannel;
};

struct SSHChannelListenerImpl : public SSHChannelListener, public kj::Refcounted {
	SSHChannelListenerImpl(Own<SSHSessionImpl> parent, LIBSSH2_LISTENER* pListener, int port);
	~SSHChannelListener();
	
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
	SSHSessionImpl(Own<AsyncIOStream> stream);
	~SSHSessionImpl();
	
	Own<SSHSessoinImpl> addImplRef() { return kj::addRef(*this); }
	
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
	void close() override ;
	bool isOpen() override ;
	Promise<void> drain() override ;
	
	//! Underlying I/O stream
	Own<AsyncIOStream> stream;
	LIBSSH2_SESSION* session;
	
	//! Promise that resolves when all writes are performed
	Promise<void> writesFinished = READY_NOW;
	
	//! Promise that tracks the currently active read
	Maybe<Promise<void>> activeRead;
	
	//! Promise holding the keep-alive loop
	Promise<void> keepAliveTask;
	
	Maybe<size_t> bytesReady;
	size_t bytesConsumed = 0;
	kj::Array<kj::byte> readBuffer;

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
		return (ssize_t) session -> writeToStream(buffer, length));
	} catch(kj::Exception& e) {
		if(e.getType() == kj::Exception::Type::DISCONNECTED) {
			return LIBSSH2_ERROR_SOCKET_DISCONNECT;
		}
	}
	
	return LIBSSH2_ERROR_SOCKET_NONE;
}

ssize_t SSHSession::receiveCallback(
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
		KJ_IF_MAYBE(pSize, session -> tryReadFromStream(buffer, length)) {
			return (ssize_t) *pSize;
		} else {
			return LIBSSH2_ERROR_EAGAIN;
		}
	} catch(kj::Exception& e) {
		if(e.getType() == kj::Exception::Type::DISCONNECTED) {
			return LIBSSH2_ERROR_SOCKET_DISCONNECT;
		}
	}
	
	return LIBSSH2_ERROR_SOCKET_NONE;
}

Promise<Own<SSHSession>> createSSHSession(Own<AsyncIOStream> stream) {
	auto session = kj::refcounted<SSHSessionImpl>(mv(stream));
	
	Promise<void> whenReady = session -> start();
	return whenReady
	.then([session = mv(session)]() mutable {
		session -> keepAliveTask = session.keepAlive().eagerlyEvaluate(nullptr);
		return mv(session);
	});
}
	

// === class SSHSessionIpml ===

namespace {
	struct NullErrorHandler : public kj::TaskSet::ErrorHandler {
		void taskFailed(kj::Exception&& exception) override {};
		
		static NullErrorHandler INSTANCE;
	};
	
	NullErrorHandler NullErrorHandler::INSTANCE;
}

SSHSessionImpl::SSHSessionImpl(Own<kj::AsyncIOStream> stream) :
	stream(mv(stream)),
	session(libssh2_session_init_ex(nullptr, nullptr, nullptr, reinterpret_cast<void*>(this))),
	tasks(NullErrorHandler::INSTANCE)
{
	// Set send and receive callbacks
	libssh2_session_callback_set(session, LIBSSH2_CALLBACK_SEND, &libssh2SendCallback);
	libssh2_session_callback_set(session, LIBSSH2_CALLBACK_RECV, &libssh2RecvCallback);
	
	// Set session to non-blocking operation
	libssh2_session_set_blocking(session, 0);
}

SSHSessionImpl::~SSHSessionImpl() {
	// Terminate the keep-alive loop
	keepAliveTask = READY_NOW;
	
	// Remove the stream to cause all send / receive calls to fail
	// Note that this is safe because libssh2SendCallback and libssh2RecvCallback
	// explicitly check whether the stream is null (don't want an exception to be
	// propagated in that case).
	stream = Own<AsyncIOStream>();
	
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
}

Promise<void> SSHSessionImpl::startup() {
	return queueLibssh2Call([this]() {
		return libssh2_session_startup(session, 0);
	});
}

Promise<void> SSHSessionImpl::keepAlive() {
	return queueOp<int>([this]() -> Maybe<int> {
		int* secondsToNext = 0;
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
		KJ_REQUIRE(activeRead == nullptr);
		
		size_t bytesToCopy = std::min(*pBytes - bytesConsumed, length);
		memcpy(buffer, readBuffer.begin(), bytesToCopy);
		bytesConsumed += bytesCopy;
		
		if(bytesConsumed >= *pBytes) {
			bytesConsumed = 0;
			bytesReady = nullptr;
		}
		
		return bytesToCopy;
	}
	
	if(activeRead == nullptr) {
		if(readBuffer.size() < length)
			readBuffer = kj::heapArray<kj::byte>(length);
		
		activeRead = stream->read(readBuffer.begin(), 1, length)
		.then([this](size_t actualBytesRead) {
			bytesReady = actualBytesRead;
			bytesConsumed = 0;
			checkAllOps();
		});
	}
	
	return nullptr;
}

ssize_t SSHSessionImpl::writeToStream(const void* buffer, size_t length) {	
	auto tmpBuf = kj::heapArray<kj::byte>(length);
	memcpy(tmpBuf.begin(), buffer, length);
	
	writesFinished = writesFinished.then([this]() {
		stream->write(tmpBuf.begin(), tmpBuf.size());
	});
	return length;
}

void SSHSession::checkAllOps() {
	for(QueuedOp& op : ops) {
		if(libSession == nullptr) {
			op.kill();
			ops.remove(op);
			continue;
		}
		
		if(op.check()) {
			op.kill();
			ops.remove(op);
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
		
		bool check() override {
			bool finished = false;
			
			if(cancelable && !fulfiller.isWaiting())
				return true;
			
			bool ok = fulfiller.rejectIfThrows([this, &finished]() {
				KJ_IF_MAYBE(pResult, op()) {
					fulfiller -> fulfill(mv(*pResult));
					finished = true;
				}
			});
			
			if(!ok) {
				finished = true;
			}
			
			return finished;
		}
		
		void kill() override {
			fulfiller.reject("Session closed");
		}
		
		Adapter(PromiseFulfiller<T>& fulfiller, kj::Function<Maybe<T>()> op, SSHSession& session, bool cancelable) :
			op(mv(op)),
			fulfiller(fulfiller),
			session(session),
			cancelable(cancelable)
		{
			session.queuedOps.add(*this);
		}
		
		~Adapter() {
			if(opsLink.isLinked()) {
				fulfiller.reject("Operation deleted");
				session.queuedOps.remove(*this);
			}
		}
	};
	
	return kj::newAdaptedPromise<Adapter>(mv(op), *this, cancelable);
}

Promise<void> queueLibssh2Call(kj::Function<int()> op) {
	auto wrapped = [op = mv(op)]() -> Maybe<int> {
		T callResult = op();
		
		if(callResult == LIBSSH2_ERROR_EAGAIN)
			return nullptr;
		
		return callResult;
	};
	
	return queueOp<T>(mv(op))
	.then([](int result) {
		KJ_REQUIRE(result == 0, "Error in LibSSH2 call");
	});
}

// --- Session interface ---

Promise<Own<SSHChannel>> SSHSessionImpl::connectRemote(kj::StringPtr remoteHost, size_t remotePort) {
	return queueOp<Own<SSHChannel>>([this, remoteHost = kj::heapString(remoteHost), remotePort] mutable -> Maybe<Own<SSHChannel>> {
		LIBSSH2_CHANNEL* pChannel = libssh2_channel_direct_tcpip(session, remoteHost.cStr(), remotePort);
		
		if(pChannel == nullptr) {
			auto errCode = libssh2_session_last_errno(session);
			KJ_REQUIRE(errCode == LIBSSH2_ERROR_EAGAIN, "libssh2_channel_direct_tcpip failed", remoteHost, remotePort);
			
			return nullptr;			
		}
		
		return kj::refcounted<SSHChannelImpl>(*this, pChannel);
	});
};

Promise<Own<SSHChannel>> SSHSessionImpl::connectRemote(kj::StringPtr remoteHost, size_t remotePort, kj::StringPtr srcHost, size_t srcPort) {
	return queueOp<Own<SSHChannel>>([this, remoteHost = kj::heapString(remoteHost), srcHost = kj::heapString(srcHost), srcPort, remotePort] mutable -> Maybe<Own<SSHChannel>> {
		LIBSSH2_CHANNEL* pChannel = libssh2_channel_direct_tcpip_ex(session, remoteHost.cStr(), remotePort, srcHost.cStr(), srcPort);
		
		if(pChannel == nullptr) {
			auto errCode = libssh2_session_last_errno(session);
			KJ_REQUIRE(errCode == LIBSSH2_ERROR_EAGAIN, "libssh2_channel_direct_tcpip_ex failed", remoteHost, remotePort, srcHost, srcPost);
			
			return nullptr;			
		}
		
		return kj::refcounted<SSHChannelImpl>(*this, pChannel);
	});
};

Promise<Own<SSHChannelListener>> listen(kj::StringPtr host, Maybe<int> port) {
	return queueOp<Own<SSHChannelListener>>([this, host = kj::heapString(host), port] mutable -> Maybe<Own<SSHChannelListener>> {
		int& boundPort = 0;
		
		int portArg = 0;
		KJ_IF_MAYBE(pPort, port) {
			portArg = *pPort;
		}
		
		// LIBSSH2_LISTENER * libssh2_channel_forward_listen_ex(LIBSSH2_SESSION *session, char *host, int port, int *bound_port, int queue_maxsize); 
		LIBSSH2_LISTENER* pListener =  libssh2_channel_forward_listen_ex(session, host.cStr(), port, &boundPort, 5);
		
		if(pChannel == nullptr) {
			auto errCode = libssh2_session_last_errno(session);
			KJ_REQUIRE(errCode == LIBSSH2_ERROR_EAGAIN, "libssh2_channel_direct_tcpip failed", host, port);
			
			return nullptr;			
		}
		
		return kj::refcounted<SSHChannelListenerImpl>(*this, pListener, boundPort);
	});
};

void SSHSessionImpl::close() {
	if(libSession == nullptr)
		return false;
	
	kj::Vector<Promise<void>> channelCloseOps;
	
	for(auto& listener : listeners) {
		listener.close();
	}
	
	for(auto& channel : channels) {
		channel.close():
	}
	
	tasks.add(
		queueLibssh2Call([this]() mutable {
			return libssh2_session_disconnect(libSession, "Disconnected by client");
		}, true)
		.then([this]) mutable {
			return queueLibssh2Call([this]() mutable {
				int result = libssh2_session_free(libSession);
				if(result == 0) libSession = nullptr;
				return result;
			}, true);
		})
	);
	tasks.add(mv(result));
}

bool SSHSessionImpl::isOpen() {
	return liBSession != nullptr;
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

SSHChannelListenerImpl::~SSHChannelListenerImpl() {
	close();
}

Promise<Own<SSHChannel>> SSHChannelListenerImpl::accept() {	
	return queueOp<Own<SSHChannel>>([this]() mutable -> Maybe<Own<SSHChannel>> {
		KJ_REQUIRE(isOpen(), "Listener was closed");
		LIBSSH2_CHANNEL* result = libssh2_channel_forward_accept(listener);
		
		if(result == nullptr) {
			auto errCode = libssh2_session_last_errno(session);
			KJ_REQUIRE(errCode == LIBSSH2_ERROR_EAGAIN, "libssh2_channel_forward_accept", port);
			
			return nullptr;			
		}
		
		return kj::refcounted<Own<SSHChannel>>(*parent, result);
	}).attach(addRef());
}

void SSHChannelListenerImpl::close() {
	if(listenersLink.isLinked()) {
		parent -> listeners.remove(*this);
		parent -> tasks.add(queueLibssh2Call([listener]() {
			 return libssh2_channel_forward_cancel(listener);
		}));
	}
}

bool SSHChannelListenerImpl::isOpen() {
	return listenersLink().isLinked();
}

Own<AsyncIOStream> SSHChannel::getStream(size_t id) {
	auto result = kj::heap<ChannelStream>(*this, id);
	channels.add(*result);
	return result;
}

// class SSHChannelImpl

SSHChannelImpl::SSHChannelImpl(Own<SSHSessionImpl> newParent, LIBSSH2_CHANNEL* channel) :
	parent(mv(newParent)),
	channel(channel)
{
	parent -> channels.add(*this);
}

SSHChannelImpl::~SSHChannelImpl() {
	close();
}

void SSHChannelImpl::close() {
	if(!isOpen())
		return;
	
	parent.tasks.add(parent -> queueLibssh2Call([channe]() {
		return libssh2_channel_close(channel);
	}).attach(addRef()));
	
	for(auto& stream : streams) {
		stream.close();
	}
}

bool SSHChannelImpl::isOpen() {
	return channelsLink.isLinked();
}

Own<AsyncIOStream> SSHChannelImpl::openStream(size_t id) {
	return kj::heap<SSHChannelStream>>(*this, id);
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

Promise<size_t> SSHChannelStream::tryRead(void* buffer, size_t minBytes, size_t maxBytes) override {	
	return parent -> session -> queueOp<size_t>([this, buffer = (unsigned char*) buffer, minBytes, maxBytes, bytesRead = (size_t) 0]() mutable -> Maybe<size_t> {
		KJ_REQUIRE(streamsLink.isLinked(), "Stream closed");
		KJ_REQUIRE(parent -> isOpen(), "Channel closed");
		
		// Try to read
		ssize_t rc = libssh2_channel_read_ex(parent -> channel, streamID, buffer, maxBytes);
		
		// No packets in queue, please ask again later
		if(rc == LIBSSH2_ERROR_EAGAIN)
			return nullptr;
		
		// If the channel was already closed, return whatever bytes we have
		if(rc == LIBSSH2_CHANNEL_CLOSED)
			return bytesRead;
				
		KJ_REQUIRE(rc >= 0, "Failure during read", streamID);
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
			
			KJ_FAIL_REQUIRE("libssh2_channel_eof failed", rc, streamID);
		} else {
			// We collected enough bytes for now.
			return bytesRead;
		}
	});
}

Promise<void> SSHChannelStream::write(void* buffer, size_t nBytes) override {
	return parent -> session -> queueOp<bool>([this, buffer = (unsigned char*) buffer, nBytes]() mutable -> Maybe<bool> {
		KJ_REQUIRE(streamsLink.isLinked(), "Stream closed");
		KJ_REQUIRE(parent -> isOpen(), "Channel closed");
			
		ssize_t rc = libssh_channel_write_ex(parent -> channel, streamID, buffer, nBytes);
		
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

Promise<void> SSHChannelStream::write(ArrayPtr<const ArrayPtr<const byte>> pieces) override {	
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
	
	if(parent -> writingStreams().empty()) {
		parent -> parent -> tasks.add(parent -> session -> queueLibssh2Call([this]() mutable {
			KJ_REQUIRE(streamsLink.isLinked(), "Stream closed");
			KJ_REQUIRE(parent -> isOpen(), "Channel closed");
			
			return libssh2_channel_send_eof(parent -> channel);
		}));
	}
}

}