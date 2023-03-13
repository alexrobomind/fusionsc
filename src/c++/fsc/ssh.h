namespace fsc {
	
struct SSHSession;

struct SSHChannel {
	virtual Promise<bool> close() = 0;
	Own<AsyncIOStream> getStream(size_t id) = 0;
	
	virtual ~SSHChannel() {}
};

struct SSHChannelListener {
	virtual Promise<Own<SSHChannel>> accept() = 0;
	
	const int portNumber;
	
	inline SSHChannelListener(int portNumber) : portNumber(portNumber) {}
};

struct SSHSession {
	Promise<bool> close() = 0;
	
	Own<SSHChannel> connectRemote(kj::StringPtr remoteHost, size_t remotePort) = 0;
	Own<SSHChannel> connectRemote(kj::StringPtr remoteHost, size_t remotePort, kj::StringPtr srcHost, size_t srcPort) = 0;
	Own<SSHChannelListener> listen(kj::StringPtr host = "0.0.0.0"_kj, Maybe<int> port = nullptr);
	
	virtual ~SSHSession() {}
};

struct SSHSession {
	SSHSession(Own<AsyncIOStream> stream);
		
	static ssize_t sendCallback(libssh2_socket_t sockfd, const void *buffer, size_t length, int flags, void **abstract);
	static ssize_t receiveCallback(libssh2_socket_t sockfd, const void *buffer, size_t length, int flags, void **abstract);
	
private:
	//! Check all ops for completion
	void checkAllOps();
	
	ssize_t tryReadFromStream(void* buffer, size_t length);
	ssize_t tryWriteToStream(const void* buffer, size_t length);

	template<typename T>
	Promise<T> runAsync(kj::Function<Maybe<T>()> op);
	
	// Helper class that keeps track of unfinished operations
	struct QueuedOp {
		kj::ListLink<QueuedOp> opsLink;
		
		virtual bool check() = 0;
		virtual void kill() = 0;
		inline virtual ~QueuedOp() {};
	};
	kj::List<QueuedOp, &QueuedOp::opsLink> ops;
	
	// Promise that resolves when all writes are performed
	Promise<void> writesFinished = READY_NOW;
	
	// Promise that tracks the currently active read
	Maybe<Promise<void>> activeRead;
	
	Maybe<size_t> bytesReady;
	size_t bytesConsumed = 0;
	kj::Array<kj::byte> readBuffer;

	LIBSSH2_SESSION* libSession;
	Own<AsyncIOStream> stream;
};
	

// === struct SSHSession ===

template<typename T>
Promise<T> SSHSession::runAsync(kj::Function<Maybe<T>()> op, bool cancelable) {
	struct Adapter : public QueuedOp {
		kj::Function<Maybe<T>()> op;
		kj::PromiseFulfiller<T>& fulfiller;
		SSHSession& session;
		
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
			session.ops.add(*this);
		}
		
		~Adapter() {
			if(opsLink.isLinked()) {
				fulfiller.reject("Operation deleted");
				session.ops.remove(*this);
			}
		}
	};
	
	return kj::newAdaptedPromise<Adapter>(mv(op), *this, cancelable);
}


}