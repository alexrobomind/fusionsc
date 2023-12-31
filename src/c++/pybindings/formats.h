#pragma once

#include "common.h"
#include "assign.h"
#include "capnp.h"

#include <fsc/textio.cpp>

namespace fscpy {
	namespace formats {
		struct Formatted : public Assignable {
			Format& parent;
			Own<kj::InputStream&> src;
			py::object input;
			
			bool used = false;
			
			inline void assign(const BuilderSlot& dst) override {
				KJ_REQUIRE(!used, "Can only assign from a formatted load object once");
				parent.read(dst, *src);
				used = true;
			}
		};
		
		struct Format {
			Format(bool binary) : isBinary(binary) {}
			
			virtual void write(DynamicValueReader, kj::BufferedOutputStream&, const textio::WriteOptions& = textio::WriteOptions()) = 0;
			virtual void read(const BuilderSlot&, kj::BufferedInputStream&) = 0;
			
			py::object dumps(DynamicValueReader, bool compact);
			void dump(DynamicValueReader, py::object, bool compact);
			
			Formatted load(py::object);
			Formatted loads1(py::buffer);
			Formatted loads2(py::str);
			
			const bool isBinary;
		};
		
		struct TextIOFormat : public Format {
			textio::Dialect dialect;
			
			TextIOFormat(const textio::Dialect&);
			
			void write(DynamicValueReader, kj::BufferedOutputStream&) override;
			void read(const BuilderSlot&, kj::BufferedInputStream&) override;
		};
		
		struct YAML : public TextIOFormat {
			YAML();
		};
		
		struct BSON : public TextIOFormat {
			BSON();
		};
		
		struct JSON : public TextIOFormat {
			JSON();
		};
		
		struct CBOR : public TextIOFormat {
			CBOR();
		};
	}
}
