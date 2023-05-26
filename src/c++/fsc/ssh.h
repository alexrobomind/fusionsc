#pragma once

#include "common.h"

#include <kj/async-io.h>

namespace fsc {

struct SSHChannel {
	virtual ~SSHChannel();
	virtual Own<SSHChannel> addRef() = 0;
	
	virtual Own<kj::AsyncIoStream> openStream(size_t id) = 0;
	
	virtual void close() = 0;
	virtual bool isOpen() = 0;
};

struct SSHChannelListener {
	virtual ~SSHChannelListener();
	
	virtual Promise<Own<SSHChannel>> accept() = 0;
	virtual int getPort() = 0;
	
	virtual Own<SSHChannelListener> addRef() = 0;
	
	virtual void close() = 0;
	virtual bool isOpen() = 0;
};

struct SSHSession {
	virtual ~SSHSession();
	virtual Own<SSHSession> addRef() = 0;
	
	virtual Promise<Own<SSHChannel>> connectRemote(kj::StringPtr remoteHost, size_t remotePort) = 0;
	virtual Promise<Own<SSHChannel>> connectRemote(kj::StringPtr remoteHost, size_t remotePort, kj::StringPtr srcHost, size_t srcPort) = 0;
	virtual Promise<Own<SSHChannelListener>> listen(kj::StringPtr host = "0.0.0.0"_kj, Maybe<int> port = nullptr) = 0;
	
	virtual Promise<bool> authenticatePassword(kj::StringPtr user, kj::StringPtr password) = 0;
	virtual Promise<bool> authenticatePubkeyFile(kj::StringPtr user, kj::StringPtr pubkeyFile, kj::StringPtr privkeyFile, kj::StringPtr passPhrase = nullptr) = 0;
	virtual Promise<bool> authenticatePubkeyData(kj::StringPtr user, kj::StringPtr pubkeyData, kj::StringPtr privkeyData, kj::StringPtr passPhrase) = 0;
	virtual bool isAuthenticated() = 0;
	
	virtual void close() = 0;
	virtual bool isOpen() = 0;
	
	virtual Promise<void> drain() = 0;
};

Promise<Own<SSHSession>> createSSHSession(Own<kj::AsyncIoStream> stream);

}