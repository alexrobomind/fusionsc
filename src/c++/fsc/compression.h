#include <zlib.h>

#include "common.h"

/** Simple interface module for ZLib */

namespace fsc {

/** Helper class for implementing ZLib-based compression and decompression */
struct ZLib {
	enum State {
		NO_PROGRESS, //! Indicates that no progress could be made because input or output buffer is full
		PROGRESS,    //! Indicates that compression / decompression progressed as planned
		FINISHED     //! Indicates that the stream is finished
	};
	
	//! Sets the input buffer to read from
	inline void setInput(kj::ArrayPtr<const byte> newInput);
	
	//! Sets the output buffer to write to.
	inline void setOutput(kj::ArrayPtr<byte> newOutput);
	
	//! Number of bytes remaining in input buffer (is == 0 if input fully consumed)
	inline size_t remainingIn();
	
	//! Number of bytes remaining in output buffer.
	inline size_t remainingOut();

protected:	
	ZLib();
	z_stream stream;
};

struct Compressor : public ZLib {
	Compressor(int level);
	~Compressor();
	
	//! Performs compression until no more input or output bytes are available.
	State step(bool finish);
};

struct Decompressor : public ZLib {
	Decompressor();
	~Decompressor();
	
	//! Performs decompression until no more input or output bytes are available.
	State step();

private:	
	bool ready = false;
};

}

// ======================================= Inline implementation =====================================

namespace fsc {

void ZLib::setInput(kj::ArrayPtr<const byte> newInput) {
	stream.next_in = const_cast<byte*>(newInput.begin());
	stream.avail_in = newInput.size();
}

void ZLib::setOutput(kj::ArrayPtr<byte> newOutput) {
	stream.next_out = newOutput.begin();
	stream.avail_out = newOutput.size();
}

size_t ZLib::remainingIn() {
	return stream.avail_in;
}
size_t ZLib::remainingOut() {
	return stream.avail_out;
}

}