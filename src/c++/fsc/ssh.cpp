namespace fsc {
	
struct SSHChannelImpl;
struct SSHChannelListenerImpl;
struct SSHChannelStream;
struct SSHSessionImpl;

struct SSHChannelStream : public AsyncIOStream {
	// AsyncInputStream
	Promise<size_t> tryRead(void* buffer, size_t minBytes, size_t maxBytes) override;
	
	// AsyncOutputStream
	Promise<void> write(void* buffer, size_t nBytes) override;
	Promise<void> write(ArrayPtr<const ArrayPtr<const byte>> pieces) override;
	
	inline ChannelStream(SSHChannel& channel, size_t streamID) :
		channel(channel), streamID(streamID)
	{}
	
	~SSHChannelStream() {};
	
private:
	kj::ListLink<ChannelStream> streamsLink;
	SSHChannelImpl& channel;
	size_t streamID;
	
	kj::Canceler canceler;
};

struct SSHChannelImpl : public SSHChannel {
	Promise<bool> close() override;
	
	Own<AsyncIOStream> getStream(size_t id);
		
	SSHSession& session;
	kj::ListLink<SSHChannel> channelsLink;
	
	LIBSSH2_CHANNEL* libChannel;
	
	List<ChannelStream, &ChannelStream::streamsLink> streams;
};

struct SSHChannelImpl {
};

ssize_t SSHSession::tryReadFromStream(void* buffer, size_t length) {
	KJ_REQUIRE(stream != nullptr);
	
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
			update();
		});
	}
	
	return -11; // -EAGAIN
}

ssize_t SSHSession::tryWriteToStream(const void* buffer, size_t length) {
	KJ_REQUIRE(stream != nullptr);
	
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

ssize_t SSHSession::sendCallback(
	libssh2_socket_t sockfd,
	const void *buffer,
	size_t length,
	int flags,
	void **abstract
) {
	SSHSession* session = (SSHSession*) *abstract;
	
	try {
		return session -> tryWriteToStream(buffer, length);
	} catch(kj::Exception& e) {
	}
	
	return -1;
}

ssize_t SSHSession::receiveCallback(
	libssh2_socket_t sockfd,
	const void *buffer,
	size_t length,
	int flags,
	void **abstract
) {
	SSHSession* session = (SSHSession*) abstract;
	
	try {
		return session -> tryReadFromStream(buffer, length);
	} catch(kj::Exception& e) {
	}
	
	return -1;
}

Promise<bool> SSHSession::close() {
	if(libSession == nullptr)
		return false;
	
	kj::Vector<Promise<void>> channelCloseOps;
	
	for(auto& channel : channels) {
		channelCloseOps.add(channel.close().ignoreResult());
	}
	
	auto result = kj::joinPromises(channelCloseOps.releaseAsArray())
	.then([this]) {
		return this -> runAsync<bool>([this]() -> Maybe<bool> {
			int returnCode = libssh2_session_disconnect(libSession, "Disconnected by client");
			
			if(returnCode == LIBSSH2_ERROR_EAGAIN)
				return nullptr;
			
			KJ_REQUIRE(returnCode == 0);
			
			return true;
		}, true);
	})
	.then([this]) {
		return this -> runAsync<bool>([this]() -> Maybe<bool> {
			int returnCode = libssh2_session_free(libSession);
			
			if(returnCode == LIBSSH2_ERROR_EAGAIN)
				return nullptr;
			
			KJ_REQUIRE(returnCode == 0);
			
			libSession = nullptr;
			return true;
		}, true);
	})
	.ignoreResult();
}

SSHSession::~SSHSession() {
	// Shoot down IO stream
	// This ensures we don't get async wait responses
	stream = nullptr;
	
	// Get rid of session
	if(libSession != nullptr) {
		libssh2_session_free(libSession);
		libSession = nullptr;
	}
	
	// Freeing the session also cleared all channels
	// Make sure that all channels are notified of that
	for(auto& channel : channels) {
		channel.libChannel = nullptr;
	}
}

Own<AsyncIOStream> SSHChannel::getStream(size_t id) {
	auto result = kj::heap<ChannelStream>(*this, id);
	channels.add(*result);
	return result;
}

Promise<bool> SSHChannel::close() {
	if(libChannel == nullptr)
		return false;
	
	auto closeChannelOp = [libChannel, this]() -> Maybe<bool> {
		int returnCode = libssh2_channel_free(libChannel);
		
		if(returnCode == LIBSSH2_ERROR_EAGAIN)
			return nullptr;
		
		KJ_REQUIRE(returnCode == 0);
		
		return true;
	};
	
	libChannel = nullptr;
	
	return session.runAsync(mv(closeChannelOp), false);
}

SSHChannel::ChannelStream::~ChannelStream() {
	if(streamsLink.isLinked()) {
		channel.streams.remove(*this);
	}
}

Promise<size_t> SSHChannel::ChannelStream::tryRead(void* buffer, size_t minBytes, size_t maxBytes) override {
	struct OpImpl {
		ChannelStream& stream;
		
		void* buffer;
		size_t minBytes;
		size_t maxBytes;
		OpImpl(ChannelStream& stream, void* buffer, size_t minBytes, size_t maxBytes) : channel(channel), buffer(buffer), minBytes(minBytes), maxBytes(maxBytes) {}

		size_t bytesRead = 0;
		
		Maybe<size_t> operator() {
			KJ_REQUIRE(stream.streamsLink -> isLinked(), "Channel deleted");
			KJ_REQUIRE(stream.channel.libChannel != nullptr, "Channel closed");
			
			ssize_t rc = libssh2_channel_read_ex(stream.channel.libChannel, stream.streamID, ((unsigned char*) buffer) + bytesRead, maxBytes - bytesRead);
			
			if(rc == LIBSSH2_ERROR_EAGAIN)
				return nullptr;
						
			KJ_REQUIRE(rc >= 0);
			
			bytesRead += rc;
			
			if(bytesRead >= minBytes)
				return bytesRead;
			else
				return nullptr;
		}
	};
	
	return canceler.wrap(session.runAsync<size_t>(OpImpl(*this, buffer, minBytes, maxBytes), true));
}

Promise<void> SSHChannel::ChannelStream::write(void* buffer, size_t nBytes) override {
	struct OpImpl {
		ChannelStream& stream;
		
		void* buffer;
		size_t maxBytes;
		OpImpl(SSHChannel& channel, void* buffer, size_t maxBytes) : channel(channel), buffer(buffer), maxBytes(maxBytes) {}
		
		size_t bytesWritten = 0;
		
		Maybe<bool> operator() {
			KJ_REQUIRE(stream.streamsLink -> isLinked(), "Channel deleted");
			KJ_REQUIRE(stream.channel.libChannel != nullptr, "Channel closed");
			
			ssize_t rc = libssh_channel_write_ex(stream.channel.libChannel, stream.streamID, ((unsigned char*) buffer) + bytesWritten, maxBytes - bytesWritten);
			
			if(rc == LIBSSH2_ERROR_EAGAIN)
				return nullptr;
			
			KJ_REQUIRE(rc > 0);
			bytesWritten += rc;
			
			if(bytesWritten >= maxBytes)
				return true;
			
			return nullptr;
		}
	};
	
	return session.runAsync<size_t>(OpImpl(*this, buffer, nBytes), true).ignoreResult();
}

Promise<void> SSHChannel::ChannelStream::write(ArrayPtr<const ArrayPtr<const byte>> pieces) override {	
	Promise<void> result = READY_NOW;
	
	for(auto piece : pieces) {
		result = result.then([this, piece]() {
			write(piece.begin(), piece.size());
		});
	}
	
	return result;
}

}