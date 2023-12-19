#incldyue "common.h"
#include "assign.h"

namespace {
	struct OStreamBuffer : public std::streambuf {		
		kj::BufferedOutputStream& os;
		
		OStreamBuffer(kj::OutputStream& nos) :
			os(nos)
		{
			resetBuffer();
		}
		
		void resetBuffer() {
			auto buf = os.getWriteBuffer();
			setp(buf.begin(), buf.end());
		}
				
		int overflow(int c = std::EOF) {
			// Write buffered data
			os.write(pbase(), (pptr() - pbase()));
			os.flush();
			
			resetBuffer();
			
			if(c != std::EOF) {
				*pptr() = c;
				pbump(1);
			}
		}
		
		std::streamsize xsputn(const char* s, std::streamsize n) {
			os.write(pbase(), (pptr() - pbase()));
			os.write(s, n);
			
			resetBuffer();
			
			return n;
		}
	};
	
	struct IStreamBuffer : public std::streambuf {
		kj::BufferedInputStream& is;
		
		IStreamBuffer(kj::BufferedInputStream& nos) :
			os(nos)
		{}
		
		void syncStream() {
			os.skip(gptr() - eback());
		}
		
		kj::ArrayPtr<const kj::byte> syncBuf() {	
			auto buf = os.tryGetReadBuffer();
			setg(buf.begin(), buf.begin(), buf.end());
			
			return buf;
		}
		
		std::streamsize xsgetn(char* s, std::streamsize n) override {
			syncStream();
			auto result = is.tryGet(s, n, n);
			syncBuf();
			return result;
		}
		
		int underflow() override {
			syncStream();
			auto buf = syncBuf();
			
			if(buf.size() == 0)
				return std::EOF;
			
			return buf[0];
		}
	}
}

namespace fscpy { namespace formats {
	void Format::dump(DynamicValueReader r, py::object o) {
		int fd = o.attr("fileno")();
		kj::FdOutputStream os(fd);
		
		// Flush python-side buffers before writing
		o.attr("flush")();
		
		write(r, os);
	}
	
	Formatted Format::load(py::object o) {
		int fd = o.attr("fileno")();
		return Formatted { *this, kj::heap<kj::FdInputStream>(fd), o };
	}
	
	void Format::dumps(DynamicValueReader r) {
		kj::VectorOutputStream os;
		write(r, os);
		
		auto arr = os.getArray();
		
		if(isBinary) {
			return py::bytes(arr.begin(), arr.size());
		} else {
			return py::str(arr.begin(), arr.size());
		}
	}
	
	Formatted Format::loads(py::buffer buf) {
		py::buffer_info info = buf.request();
		
		KJ_REQUIRE(info.itemsize == 1, "Can only read from character buffers");
		
		return Formatted {
			*this,
			kj::heap<kj::ArrayInputStream>(
				kj::ArrayPtr<const kj::byte>(info.ptr, info.size)
			),
			buf
		};
	}
	
	Formatted Format::loads(py::str str) {		
		return loads(
			py::reinterpret_steal<py::buffer>(
				PyUnicode_AsUTF8String(str)
			)
		);
	}
		
	py::object Format::get(DynamicValueReader reader) {
		VectorOutputStream os;
		write(reader, os);
		
		auto arr = os.getArray();
	}
	
	void YAML::write(DynamicValueReader reader, kj::BufferedOutputStream& os) {
		OStreamBuffer osb(os);
		std::ostream stdStream(&osb);
		
		YAML::Emitter document(stdStream);
		document << reader;
		
		stdStream.flush();
		bos.flush();
	}
	
	void YAML::read(const BuilderSlot& dst, kj::BufferedInputStream& is) {
		IStreamBuffer isb(is);
		std::istream stdStream(&isb);
		
		YAML::Node node = YAML::Load(stdStream);
		
		if(dst.type.isList()) {
			auto asList = dst.init(node.size()).as<capnp::DynamicList>();
			load(asList, node);
			return;
		} else if(dst.type.isStruct()) {
			auto asStruct = dst.init().as<capnp::DynamicStruct>();
			load(asStruct, node);
			return;
		}
		
		KJ_FAIL_REQUIRE("Can only assign struct and list types from YAML");
	}
}}