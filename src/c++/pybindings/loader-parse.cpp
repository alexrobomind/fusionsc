#include "fscpy.h"
#include "loader.h"

#include <list>

#include <capnp/schema-parser.h>

namespace fscpy {

namespace {

struct SourceLoader : public kj::Refcounted {
	using SourcePos = capnp::SchemaFile::SourcePos;
	
	py::dict roots;
	
	SourceLoader(py::dict roots) :
		roots(mv(roots))
	{}
	
	void check() {
		if(errors.empty())
			return;
		
		for(Error& e : errors) {
			py::print(kj::str("Error in file ", e.pth, ", line ", e.pos1.line, ":", e.pos1.column, " to line ", e.pos2.line, ":", e.pos2.column, " - ", e.message)); 
		}
		errors.clear();
		
		KJ_FAIL_REQUIRE("Errors during schema parsing");
	}
	
	void reportError(kj::PathPtr pth, SourcePos pos1, SourcePos po2, kj::StringPtr message) {
		errors.push_back(Error { pth.clone(), pos1, pos1, kj::heapString(message) });
	}
	
	Own<SourceLoader> addRef() {
		return kj::addRef(*this);
	}
	
	Array<capnp::Schema> getAllLoaded() {
		return parser.getAllLoaded();
	}
	
	// Implemented below due to requiring PyResourceFile
	capnp::ParsedSchema parseFile(py::object pathRoot, kj::PathPtr path);
	
	Maybe<Own<capnp::SchemaFile>> openFile(py::object pathRoot, kj::PathPtr pth);
	
private:
	struct Error {
		kj::Path pth;
		SourcePos pos1;
		SourcePos pos2;
		kj::String message;
	};
	
	capnp::SchemaParser parser;
	std::list<Error> errors;
};

struct PyResourceFile : public capnp::SchemaFile {
	py::object pathRoot;
	kj::Path pth;
	
	mutable py::object pyPath;
	mutable Own<SourceLoader> loader;
	
	PyResourceFile(py::object pathRoot, kj::PathPtr pth, py::object pyPath, SourceLoader& loader) :
		pathRoot(mv(pathRoot))
		pth(pth.clone()),
		pyPath(mv(pyPath)),
		loader(loader.addRef())
	{}
	
	kj::StringPtr getDisplayName() const override {
		return pth.basename()[0];
	}
	
	kj::Array<const char> readContent() const override {
		return py::cast<kj::Array<const byte>>(pyPath.attr("read_bytes")());
	}
	
	Maybe<Own<SchemaFile>> import(kj::StringPtr path) const override {
		try {
			// Absolute paths are indicated by using "none" as root
			if(path.startsWith("/"))
				return loader -> openFile(py::none(), kj::Path().eval(path));
			
			// Parse relative path
			return loader -> openFile(pathRoot, this -> pth.parse(path));
		} catch(std::exception& e) {
			py::print("Error occurred opening file", e.what());
			return nullptr;
		}
	}
	
	bool operator==(const SchemaFile& other) const override {
		const PyResourceFile* downcast = dynamic_cast<const PyResourceFile*>(&other);
		
		if(downcast == nullptr)
			return false;
		
		if(pathRoot.ptr() != downcast -> pathRoot.ptr())
			return false;
		
		return pth == downcast -> pth;
	}
	bool operator!=(const SchemaFile& other) const override {
		return ! operator==(other);
	}
	
	size_t hashCode() const override {
		return pathRoot.ptr() + kj::hashCode(pth);
	}
	
	void reportError(SourcePos pos1, SourcePos pos2, kj::StringPtr message) const override {
		return loader -> reportError(pth, pos1, pos2, message);
	}
};

capnp::ParsedSchema SourceLoader::parseFile(py::object pathRoot, kj::PathPtr path) {
	KJ_IF_MAYBE(ppFile, openFile(path)) {
		auto result = parser.parseFile(mv(*ppFile));
		check();
		return result;
	} else {
		KJ_FAIL_REQUIRE("Could not open file", path);
	}
}

Maybe<Own<capnp::SchemaFile>> SourceLoader::openFile(py::object pathRoot, kj::PathPtr path) {
	auto current = pathRoot;
	if(current.is_none()) {
		if(path.size() == 0)
			return nullptr;
		
		auto rootName = py::cast(path[0]);
		if(!roots.contains(rootName))
			return nullptr;
		
		current = roots[rootName];
		path = path.slice(1, path.size());
	}
	
	for(kj::StringPtr el : path) {
		if(!py::bool_(current.attr("is_dir")()))
			return nullptr;
		
		current = current.attr("joinpath")(el);
	}
			
	if(!py::bool_(current.attr("is_file")))
		return nullptr;
	
	return kj::heap<PyResourceFile>(path, mv(current), *this);
}

}

void parseSchema(py::object anchor, kj::StringPtr path, py::module target, py::dict roots) {
	auto loader = kj::refcounted<SourceLoader>(roots);
	
	auto parsed = loader -> parseFile(anchor, kj::Path().parse(path));
	
	auto loaded = loader -> getAllLoaded();
	defaultLoader.add(parsed.getProto());
	
	defaultLoader.importNodeIfRoot(parsed, schema.getProto().getId(), target);
}

}