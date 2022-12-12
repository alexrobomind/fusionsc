#include "compression.h"

namespace fsc {
	
ZLib::ZLib() {
	memset(&stream, 0, sizeof(z_stream));
}

// Compressor

Compressor::Compressor(int level) {
	deflateInit(&stream, level);
}

Compressor::~Compressor() {
	deflateEnd(&stream);
}

ZLib::State Compressor::step(bool finish) {
	auto retCode = deflate(&stream, finish ? Z_FINISH : Z_NO_FLUSH);
	
	if(retCode == Z_BUF_ERROR) {
		KJ_ASSERT(remainingIn() == 0 || remainingOut() == 0, "ZLib assumption violated: Space available but no progress");
		return NO_PROGRESS;
	}
	
	if(retCode == Z_OK)
		return PROGRESS;
	
	if(retCode == Z_STREAM_END)
		return FINISHED;
	
	kj::StringPtr errorMessage(stream.msg);
	KJ_FAIL_REQUIRE("Compression error", retCode, errorMessage);
}

// Decompressor

Decompressor::Decompressor() {
	inflateInit(&stream);
}

Decompressor::~Decompressor() {
	inflateEnd(&stream);
}

ZLib::State Decompressor::step() {
	auto retCode = inflate(&stream, Z_NO_FLUSH);
	
	if(retCode == Z_BUF_ERROR) {
		KJ_ASSERT(remainingIn() == 0 || remainingOut() == 0, "ZLib assumption violated: Space available but no progress");
		return NO_PROGRESS;
	}
	
	if(retCode == Z_OK)
		return PROGRESS;
	
	if(retCode == Z_STREAM_END)
		return FINISHED;
	
	kj::StringPtr errorMessage(stream.msg);
	KJ_FAIL_REQUIRE("Decompression error", retCode, errorMessage);
}

}