#pragma once

#include <capnp/schema-loader.h>
#include <capnp/schema.h>

#include <kj/string-tree.h>

#include <fsc/data.h>
#include <fsc/typing.h>

namespace fscpy {
	struct Loader {
		/** Adds the contents of a Cap'n'proto schema node to a module.
		 *
		 * If the schema node is a struct / interface / enum / const, it will
		 * be added as an appropriately typed object to the module.
		 *
		 * If it is a file, all contained nodes will be added to the module.
		 */	
		bool addToModule(uint64_t nodeID, py::module scope);
		
		/**
		 * Adds the contents of the schema node if it isn't a nested struct or
		 * a generated method parameter / result struct.
		 */
		bool addToModuleIfAppropriate(uint64_t nodeID, py::module scope);
		
		void load(capnp::schema::Node::Reader reader);
		void loadSource(capnp::schema::Node::SourceInfo::Reader reader);
		
		/** Load a C++ type description
		 *
		 * This loads the schema node bundled with the associated Cap'n'proto C++
		 * interface class. It uses capnp::loadCompiledTypeAndDependencies,
		 * so it also correctly sets up the associations to mark dynamic types
		 * as castable to this class
		 */
		template<typename... T>
		void loadBuiltin();
		
		//! Rebuilds the input schema from types registered in here (incl. brands)
		capnp::Schema equivalentSchema(capnp::Schema input);
		
		template<typename T>
		capnp::Schema schemaFor();
		
		capnp::SchemaLoader capnpLoader;
		
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
		kj::HashMap<capnp::Schema, capnp::Schema> rebuiltSchemas;
		
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
		
		py::type makeCommonType(capnp::Type);
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
	void Loader::loadBuiltin() {
		using arrType = int [];
		(void) arrType { 0, (capnpLoader.loadCompiledTypeAndDependencies<T>(), 0)... };
	}
	
	template<typename T>
	capnp::Schema Loader::schemaFor() {
		return equivalentSchema(capnp::Schema::from<T>());
	}
	
	/** Parses schema files available through importlib-resources.
	 *
	 * @param anchor An import-lib resources object representing the current dir
	   for initial relative path resolution. May be None if the root path is
	   absolute.
	   
	   @param path Root schema path (folder or file)
	   @param target Destination scope (module) to add the data to.
	   @param roots Dict to look up root paths (str -> importlib resource objects)
	 */ 
	void parseSchema(py::object anchor, kj::StringPtr path, py::object target, py::dict roots);
	
	//! Returns a python-safe name equivalent for the input name
	kj::String memberName(kj::StringPtr);
}
