#include "fscpy.h"
#include "loader.h"

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
	
	// Implemented below due to requiring PyResourceFile
	capnp::ParsedSchema parseFile(kj::PathPtr path);
	
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
	mutable Own<SourceLoader> loader;
	
	PyResourceFile(kj::PathPtr pth, SourceLoader& loader) :
		pth(pth.clone()),
		loader(loader.addRef())
	{}
	
	kj::StringPtr getDisplayName() const override {
		return pth.basename()[0];
	}
	
	kj::Array<const char> readContent() const override {
		auto current = loader -> roots[py::cast(pth[0])];
		
		for(kj::StringPtr el : pth.slice(1, pth.size())) {
			current = current.attr("joinpath")(el);
		}
		
		auto opened = current.attr("open")();
		
		try {
			auto data = opened.attr("read")();
			opened.attr("close")();
			
			return py::cast<kj::String>(data).releaseArray();
		} catch(...) {
			opened.attr("close")();
			throw;
		}
	}
	
	Maybe<Own<SchemaFile>> import(kj::StringPtr path) const override {
		try {
			auto subPath = pth.eval(path);
			
			auto current = loader -> roots[py::cast(subPath[0])];
			for(kj::StringPtr el : subPath.slice(1, subPath.size())) {
				current = current.attr("joinpath")(el);
			}
			
			if(!py::cast<py::bool_>(current.attr("is_file")()))
				return nullptr;
			
			return kj::heap<PyResourceFile>(subPath, *loader);
		} catch(std::exception& e) {
			return nullptr;
		}
	}
	
	bool operator==(const SchemaFile& other) const override {
		const PyResourceFile* downcast = dynamic_cast<const PyResourceFile*>(&other);
		
		if(downcast == nullptr)
			return false;
		
		return pth == downcast -> pth;
	}
	bool operator!=(const SchemaFile& other) const override {
		return ! operator==(other);
	}
	
	size_t hashCode() const override {
		return pth.hashCode();
	}
	
	void reportError(SourcePos pos1, SourcePos pos2, kj::StringPtr message) const override {
		return loader -> reportError(pth, pos1, pos2, message);
	}
};

capnp::ParsedSchema SourceLoader::parseFile(kj::PathPtr path) {
	auto result = parser.parseFile(kj::heap<PyResourceFile>(path, *this));
	check();
	return result;
}

}

void parseSchemas(py::object localScope, py::module target, py::dict globalScope) {
	KJ_REQUIRE(!globalScope.contains(":"), "The scope key ':' is reserved");
	
	globalScope = globalScope.attr("copy")();
	globalScope[":"] = localScope;
	
	auto loader = kj::refcounted<SourceLoader>(globalScope);
	
	for(auto key : localScope) {
		kj::Path pth({":", py::cast<kj::String>(key)});
		auto schema = loader -> parseFile(pth);
		
		for(auto nested : schema.getAllNested()) {
			defaultLoader.import(schema);	
			defaultLoader.importNodeIfRoot(nested.getProto().getId(), target);
		}
	}
}

}