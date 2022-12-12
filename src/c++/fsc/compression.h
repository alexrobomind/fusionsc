#include <zlib.h>

#include "common.h"

namespace fsc {

struct ZLib {
	enum State {
		NO_PROGRESS,
		PROGRESS,
		FINISHED
	};
	
	inline void setInput(kj::ArrayPtr<const byte> newInput);
	inline void setOutput(kj::ArrayPtr<byte> newOutput);
	
	inline size_t remainingIn();
	inline size_t remainingOut();

protected:	
	ZLib();
	z_stream stream;
};

struct Compressor : public ZLib {
	Compressor(int level);
	~Compressor();
	
	State step(bool finish);
};

struct Decompressor : public ZLib {
	Decompressor();
	~Decompressor();
	
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