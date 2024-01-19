#pragma once

#include "fscpy.h"
#include "assign.h"
#include "capnp.h"

#include <fsc/textio.h>

namespace fscpy {
	namespace formats {
		using Language = textio::Dialect::Language;
		// struct FormattedReader;
		
		/*struct Format {			
			virtual Own<textio::Visitor> createVisitor(kj::BufferedOutputStream&, const textio::SaveOptions& = textio::SaveOptions()) = 0;
			virtual void read(textio::Visitor& dst, kj::BufferedInputStream&) = 0;
			
			py::object dumps(py::object, bool compact, bool asBytes);
			void dump(py::object, py::object, bool compact);
			
			FormattedReader open(py::object);
			
			inline virtual ~Format() {};
		};
		
		struct FormattedReader : public Assignable {
			Format& parent;
			Own<kj::BufferedInputStream> src;
			
			FormattedReader(Format& p, Own<kj::BufferedInputStream>&& nSrc) :
				parent(p), src(mv(nSrc))
			{}
			FormattedReader(FormattedReader&&) = default;
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
		};*/
		
		py::object dumps(py::object, Language, bool compact, bool asBytes);
		void dump(py::object, int, Language, bool compact);
		
		py::object read(py::object src, py::object dst);
	}
}
