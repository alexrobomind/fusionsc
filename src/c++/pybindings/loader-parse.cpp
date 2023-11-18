#include "fscpy.h"
#include "loader.h"

#include <list>

#include <capnp/schema-parser.h>

namespace fscpy {

namespace {

struct SourceLoader : public kj::Refcounted {
	using SourcePos = capnp::SchemaFile::SourcePos;
	
	py::dict roots;
	
	SourceLoader(py::dict rootScope) :
		roots(mv(rootScope))
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
	capnp::ParsedSchema parseFile(kj::PathPtr path);
	
	Maybe<Own<capnp::SchemaFile>> openFile(kj::PathPtr pth);
	
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
	kj::Path pth;
	mutable py::object pyPath;
	mutable Own<SourceLoader> loader;
	
	PyResourceFile(kj::PathPtr pth, py::object pyPath, SourceLoader& loader) :
		pth(pth.clone()),
		pyPath(mv(pyPath)),
		loader(loader.addRef())
	{}
	
	kj::StringPtr getDisplayName() const override {
		return pth.basename()[0];
	}
	
	kj::Array<const char> readContent() const override {		
		auto opened = pyPath.attr("open")();
		
		try {
			auto data = opened.attr("read")();
			opened.attr("close")();
			
			auto asArray = kj::heapString(py::cast<kj::StringPtr>(data)).releaseArray();
			
			// Get rid of 0 terminator
			return asArray.slice(0, asArray.size() - 1).attach(mv(asArray));
		} catch(...) {
			opened.attr("close")();
			throw;
		}
	}
	
	Maybe<Own<SchemaFile>> import(kj::StringPtr path) const override {
		auto subPath = pth.parent().eval(path);
		return loader -> openFile(subPath);
	}
	
	kj::String strPath() const {
		return kj::heapString(py::cast<kj::StringPtr>(py::str(pyPath)));
	}
	
	bool operator==(const SchemaFile& other) const override {
		const PyResourceFile* downcast = dynamic_cast<const PyResourceFile*>(&other);
		
		if(downcast == nullptr)
			return false;
		
		return strPath() == downcast -> strPath();
	}
	bool operator!=(const SchemaFile& other) const override {
		return ! operator==(other);
	}
	
	size_t hashCode() const override {
		return kj::hashCode(strPath());
	}
	
	void reportError(SourcePos pos1, SourcePos pos2, kj::StringPtr message) const override {
		return loader -> reportError(pth, pos1, pos2, message);
	}
};

capnp::ParsedSchema SourceLoader::parseFile(kj::PathPtr path) {
	KJ_IF_MAYBE(ppFile, openFile(path)) {
		auto result = parser.parseFile(mv(*ppFile));
		check();
		return result;
	} else {
		KJ_FAIL_REQUIRE("Could not open file", path);
	}
}

Maybe<Own<capnp::SchemaFile>> SourceLoader::openFile(kj::PathPtr path) {			
	auto current = roots[py::cast(path[0])];
	for(kj::StringPtr el : path.slice(1, path.size())) {
		if(!py::bool_(current.attr("is_dir")()))
			return nullptr;
		current = current.attr("joinpath")(el);
	}
			
	if(!py::bool_(current.attr("is_file")))
		return nullptr;
	
	return kj::heap<PyResourceFile>(path, mv(current), *this);
}

}

void parseSchemas(py::object localScope, py::module target, py::dict globalScope) {
	KJ_REQUIRE(!globalScope.contains(":"), "The scope key ':' is reserved");
	
	globalScope = globalScope.attr("copy")();
	globalScope[":"] = localScope;
	
	auto loader = kj::refcounted<SourceLoader>(globalScope);
	
	kj::Vector<capnp::ParsedSchema> parsed;
	for(auto child : localScope.attr("iterdir")()) {
		if(!py::bool_(child.attr("is_file")()))
			continue;
		
		kj::Path pth({":", py::cast<kj::StringPtr>(child.attr("name"))});
		
		if(!pth.basename()[0].endsWith(".capnp"))
			continue;
		
		parsed.add(loader -> parseFile(pth));
	}
	
	auto loaded = loader -> getAllLoaded();
	for(auto& schema : loaded) {
		defaultLoader.add(schema.getProto());
	}
	
	for(auto& schema : parsed) {
		defaultLoader.importNodeIfRoot(schema.getProto().getId(), target);
	}
}

}