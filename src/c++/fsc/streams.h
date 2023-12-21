#pragma once

#include <fsc/streams.capnp.h>
#include <kj/async-io.h>
#include <kj/io.h>

#include <iostream>

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

/** Buffers the input stream
 *
 * Creates an eagerly consuming buffer (based on linked list of blocks) that consumes
 * data from the given input stream. The internal buffer blocks are released once the
 * data is no longer required. The input stream features an optimized tee implementation
 * that shared the buffer and only forks the buffer's cursor position.
 */
Own<kj::AsyncInputStream> buffer(Own<kj::AsyncInputStream>&& in, uint64_t limit = kj::maxValue);

//! A variant of kj::OutputStream where multiple write calls may be simultaneously active.
struct MultiplexedOutputStream : public kj::AsyncOutputStream {
	virtual Own<MultiplexedOutputStream> addRef() = 0;
};

/** Multiplexes the output stream
 *
 * This returns an object that can safely allow multiple simultaneous writes to the given
 * output stream by multiplexing write calls in turn with a FIFO.
 */
Own<MultiplexedOutputStream> multiplex(Own<kj::AsyncOutputStream>&&);

//! Wraps a buffered input stream into an std::istream	
Own<std::istream> asStdStream(kj::BufferedInputStream& is);

//! Wraps a buffered output stream into an std::ostream
Own<std::ostream> asStdStream(kj::BufferedOutputStream& is);

}