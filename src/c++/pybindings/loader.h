#pragma once

#include "fscpy.h"

#include <capnp/schema-loader.h>
#include <capnp/schema.h>

#include <kj/string-tree.h>

#include <fsc/data.h>

namespace fscpy {
	struct Loader {		
		bool importNode(uint64_t nodeID, py::module scope);
		bool importNodeIfRoot(uint64_t nodeID, py::module scope);
		
		void add(capnp::schema::Node::Reader reader);
		
		template<typename... T>
		void addBuiltin();
		
		capnp::Schema import(capnp::Schema input);
		
		template<typename T>
		capnp::Schema importBuiltin();
		
		capnp::SchemaLoader capnpLoader;
		
		kj::HashMap<capnp::Schema, capnp::Schema> imported;
	};
	
	extern Loader defaultLoader;
	
	kj::StringTree typeName(capnp::Type type);
	
	template<typename... T>
	fsc::Temporary<capnp::List<capnp::schema::Node>> getBuiltinSchemas() {
		capnp::SchemaLoader loader;
		
		using arrType = int [];
		(void) arrType { 0, (loader.loadCompiledTypeAndDependencies<T>(), 0)... };
		
		auto allLoaded = loader.getAllLoaded();
		
		fsc::Temporary<capnp::List<capnp::schema::Node>> result(allLoaded.size());
		for(size_t i = 0; i < allLoaded.size(); ++i)
			result.setWithCaveats(i, allLoaded[i].getProto());
		
		return result;
	}
	
	template<typename... T>
	void Loader::addBuiltin() {
		using arrType = int [];
		(void) arrType { 0, (capnpLoader.loadCompiledTypeAndDependencies<T>(), 0)... };
	}
	
	template<typename T>
	capnp::Schema Loader::importBuiltin() {
		return import(capnp::Schema::from<T>());
	}
	
	void extractBrand(capnp::Schema in, capnp::schema::Brand::Builder out);
	void extractType(capnp::Type in, capnp::schema::Type::Builder out);
}