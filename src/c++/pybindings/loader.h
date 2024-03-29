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
		
		kj::HashMap<uint64_t, kj::Tuple<uint64_t, uint16_t>> paramOfMethod;
		kj::HashMap<uint64_t, kj::Tuple<uint64_t, uint16_t>> resultOfMethod;
		
		kj::Tuple<kj::StringPtr, kj::StringTree> qualName(capnp::Type);
		kj::Tuple<kj::StringPtr, kj::StringTree> qualName(capnp::Schema);
		kj::Tuple<kj::StringPtr, kj::StringTree> qualName(capnp::InterfaceSchema::Method);
		
		kj::String fullName(capnp::Type);
		
		py::type commonType(capnp::Type);
		
		py::type builderType(capnp::Type);
		py::type readerType(capnp::Type);
		py::type pipelineType(capnp::StructSchema);
		py::type clientType(capnp::InterfaceSchema);
		py::type serverType(capnp::InterfaceSchema);
		
		py::object makeInterfaceMethod(capnp::InterfaceSchema::Method);
	
	private:
		kj::HashMap<capnp::Type, py::type> builderTypes;
		kj::HashMap<capnp::Type, py::type> readerTypes;
		kj::HashMap<capnp::Type, py::type> commonTypes;
		kj::HashMap<capnp::StructSchema, py::type> pipelineTypes;
		kj::HashMap<capnp::InterfaceSchema, py::type> clientTypes;
		kj::HashMap<capnp::InterfaceSchema, py::type> serverTypes;
		kj::HashMap<capnp::EnumSchema, py::type> enumTypes;
		
		py::type makeBuilderType(capnp::Type);
		py::type makeBuilderType(capnp::StructSchema);
		py::type makeBuilderType(capnp::ListSchema);
		
		py::type makeReaderType(capnp::Type);
		py::type makeReaderType(capnp::StructSchema);
		py::type makeReaderType(capnp::ListSchema);
		
		py::type makePipelineType(capnp::StructSchema);
		py::type makeClientType(capnp::InterfaceSchema);
		py::type makeServerType(capnp::InterfaceSchema);
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
	
	//! Returns a python-safe name equivalent for the input name
	kj::String memberName(kj::StringPtr);
}
