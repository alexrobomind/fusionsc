namespace fsc {

struct SSHChannel {
	virtual ~SSHChannel() = 0;
	virtual Own<SSHChannel> addRef() = 0;
	
	virtual Own<AsyncIOStream> openStream(size_t id) = 0;
	
	virtual void close() = 0;
	virtual bool isOpen() = 0;
};

struct SSHChannelListener {
	virtual ~SSHChannelListener() = 0;
	
	virtual Promise<Own<SSHChannel>> accept() = 0;
	virtual int getPort() = 0;
	
	virtual Own<SSHChannelListener> addRef() = 0;
	
	virtual void close() = 0;
	virtual bool isOpen() = 0;
};

struct SSHSession {
	virtual ~SSHSession() = 0;
	Own<SSHSession> addRef() = 0;
	
	virtual Promise<Own<SSHChannel>> connectRemote(kj::StringPtr remoteHost, size_t remotePort) = 0;
	virtual Promise<Own<SSHChannel>> connectRemote(kj::StringPtr remoteHost, size_t remotePort, kj::StringPtr srcHost, size_t srcPort) = 0;
	virtual Promise<Own<SSHChannelListener>> listen(kj::StringPtr host = "0.0.0.0"_kj, Maybe<int> port = nullptr) = 0;
	
	virtual Promise<bool> authenticatePassword(kj::StringPtr user, kj::StringPtr password) = 0;
	
	virtual void close() = 0;
	virtual bool isOpen() = 0;
	
	Promise<void> drain();
};

Promise<SSHSession> createSSHSession(Own<AsyncIOStream> stream);

}