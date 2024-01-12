#pragma once

#include "fscpy.h"
#include "assign.h"
#include "capnp.h"

#include <fsc/textio.h>

namespace fscpy {
	namespace formats {
		struct FormattedReader;
		
		struct Format {			
			virtual Own<textio::Visitor> createVisitor(kj::BufferedOutputStream&, const textio::SaveOptions& = textio::SaveOptions()) = 0;
			virtual void read(textio::Visitor& dst, kj::BufferedInputStream&) = 0;
			
			py::object dumps(py::object, bool compact, bool asBytes);
			void dump(py::object, py::object, bool compact);
			
			FormattedReader open(py::object);
		};
		
		struct FormattedReader : public Assignable {
			Format& parent;
			Own<kj::BufferedInputStream> src;
			
			FormattedReader(Format& p, Own<kj::BufferedInputStream>&& nSrc) :
				parent(p), src(mv(nSrc))
			{}
			inline ~FormattedReader() noexcept {};
			
			bool used = false;
			
			void assign(const BuilderSlot& dst) override;
			py::object read(py::object dst);
		};
		
		struct TextIOFormat : public Format {
			textio::Dialect dialect;
			
			TextIOFormat(const textio::Dialect&);
			
			Own<textio::Visitor> createVisitor(kj::BufferedOutputStream&, const textio::SaveOptions&) override;
			void read(textio::Visitor& dst, kj::BufferedInputStream&) override;
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
