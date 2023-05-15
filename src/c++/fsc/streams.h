#pragma once

#include <fsc/streams.capnp.h>
#include <kj/async-io.h>

#include "common.h"

namespace fsc {
	
struct StreamConverter {
	virtual ~StreamConverter() noexcept(false) ;
	
	virtual RemoteInputStream::Client toRemote(Own<kj::AsyncInputStream>) = 0;
	virtual RemoteOutputStream::Client toRemote(Own<kj::AsyncOutputStream>) = 0;
	
	virtual Promise<Own<kj::AsyncInputStream>> fromRemote(RemoteInputStream::Client clt) = 0;
	virtual Promise<Own<kj::AsyncOutputStream>> fromRemote(RemoteOutputStream::Client clt) = 0;
};

Own<StreamConverter> newStreamConverter();

}