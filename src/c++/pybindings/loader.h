#pragma once

#include <capnp/schema-loader.h>
#include <capnp/schema.h>

#include <kj/string-tree.h>

#include <fsc/data.h>
#include <fsc/typing.h>

namespace fscpy {
	struct Loader {		
		bool importNode(uint64_t nodeID, py::module scope);
		bool importNodeIfRoot(uint64_t nodeID, py::module scope);
		
		void add(capnp::schema::Node::Reader reader);
		void addSource(capnp::schema::Node::SourceInfo::Reader reader);
		
		template<typename... T>
		void addBuiltin();
		
		capnp::Schema import(capnp::Schema input);
		
		template<typename T>
		capnp::Schema importBuiltin();
		
		capnp::SchemaLoader capnpLoader;
		
		kj::HashMap<capnp::Schema, capnp::Schema> imported;
		kj::HashMap<uint64_t, fsc::Temporary<capnp::schema::Node::SourceInfo>> sourceInfo;
		kj::HashMap<uint64_t, kj::String> rootModules;
		
		kj::Tuple<kj::StringPtr, kj::StringTree> qualName(capnp::Type);
		kj::Tuple<kj::StringPtr, kj::StringTree> qualName(capnp::Schema);
		kj::Tuple<kj::StringPtr, kj::StringTree> qualName(capnp::InterfaceSchema::Method);
		
		py::type builderType(uint64_t);
		py::type readerType(uint64_t);
		py::type pipelineType(uint64_t);
		py::type clientType(uint64_t);
	
	private:
		kj::HashMap<uint64_t, py::type> builderTypes;
		kj::HashMap<uint64_t, py::type> readerTypes;
		kj::HashMap<uint64_t, py::type> pipelineTypes;
		kj::HashMap<uint64_t, py::type> clientTypes;
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
	
	void parseSchema(py::object anchor, kj::StringPtr path, py::object target, py::dict roots);
}
