#pragma once

#include "common.h"
#include "assign.h"
#include "capnp.h"

#include <fsc/json.cpp>

namespace fscpy {
	namespace formats {
		struct Formatted : public Assignable {
			Format& parent;
			Own<kj::InputStream&> src;
			py::object input;
			
			inline void assign(const BuilderSlot& dst) override {
				parent.read(dst, *src);
			}
		};
		
		struct Format {
			Format(bool binary) : isBinary(binary) {}
			
			virtual void write(DynamicValueReader, kj::BufferedOutputStream&) = 0;
			virtual void read(const BuilderSlot&, kj::BufferedInputStream&) = 0;
			
			py::object dumps(DynamicValueReader);
			void dump(DynamicValueReader, py::object);
			
			Formatted load(py::object);
			Formatted loads(py::buffer);
			Formatted loads(py::str);
			
			const bool isBinary;
		};
		
		struct YAML : public Format {
			inline YAML() : Format(false) {}
			
			void write(DynamicValueReader, kj::BufferedOutputStream&) override;
			void read(const BuilderSlot&, kj::BufferedInputStream&) override;
		};
		
		struct JsonDialect : public Format {
			inline JsonDialect(const JsonOpts& nOpts) :
				Format(nOpts.isBinary()),
				opts(nOpts)
			{}
			
			JsonOptions opts;
			
			void write(DynamicValueReader, kj::BufferedOutputStream&) override;
			void read(const BuilderSlot&, kj::BufferedInputStream&) override;
		};
	}
}
