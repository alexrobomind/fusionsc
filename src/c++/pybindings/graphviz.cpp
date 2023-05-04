#include "graphviz.h"

#include <atomic>

using capnp::DynamicValue;
using capnp::DynamicList;
using capnp::DynamicStruct;
using capnp::DynamicEnum;
using capnp::DynamicCapability;
using capnp::AnyPointer;
using capnp::AnyList;

namespace py = pybind11;

namespace fscpy {

namespace {

uint64_t nodeUUID() {
	static thread_local uint64_t value = 0;
	return value++;
}

kj::Vector<uint64_t> addDynamicToGraph(py::object graph, capnp::DynamicValue::Reader reader);

void addGraphNode(py::object graph, uint64_t id, kj::StringPtr label) {
	graph.attr("node")(kj::str(id), label);
}
void addGraphEdge(py::object graph, uint64_t id1, uint64_t id2, kj::StringPtr label) {
	graph.attr("edge")(kj::str(id1), kj::str(id2), label);
}

bool canAddToGraph(capnp::Type type) {
	if(type.isStruct()) return true;
	if(type.isInterface()) return true;
	
	if(type.isList()) {
		auto wrapped = type.asList().getElementType();
		
		if(wrapped.isStruct()) return true;
		if(wrapped.isInterface()) return true;
		if(wrapped.isList()) return true;
	}
	
	return false;
}

uint64_t addStructToGraph(py::object graph, capnp::DynamicStruct::Reader reader) {
	auto schema = reader.getSchema();
	
	auto nodeCaption = kj::strTree(schema.getUnqualifiedName());
	
	uint64_t nodeId = nodeUUID();
	
	kj::Function<void(capnp::StructSchema::Field, capnp::DynamicStruct::Reader)> handleField;
	kj::Function<void(capnp::DynamicStruct::Reader)> handleGroup;
	
	handleField = [&](capnp::StructSchema::Field field, capnp::DynamicStruct::Reader reader) {
		auto type = field.getType();
		
		if(canAddToGraph(type)) {
			auto children = addDynamicToGraph(graph, reader.get(field));
			for(uint64_t child : children) {
				addGraphEdge(graph, nodeId, child, field.getProto().getName());
			}
		} else {
			nodeCaption = kj::strTree(mv(nodeCaption), "\n", field.getProto().getName(), " = ", reader.get(field));
		}
	};
	
	handleGroup = [&](capnp::DynamicStruct::Reader reader) {
		auto schema = reader.getSchema();
		
		uint64_t nSubGroups = 0;
		
		KJ_IF_MAYBE(pField, reader.which()) {
			if(pField -> getProto().isGroup())
				++nSubGroups;
		}
				
		for(auto field : schema.getNonUnionFields()) {
			if(field.getProto().isGroup())
				++nSubGroups;
		}
		
		KJ_IF_MAYBE(pField, reader.which()) {
			auto& field = *pField;
			if(field.getProto().isGroup() && nSubGroups <= 1) {
			} else {
				nodeCaption = kj::strTree(mv(nodeCaption), "\n", field.getProto().getName());
				handleField(field, reader);
			}
		}
				
		for(auto field : schema.getNonUnionFields()) {
			if(field.getProto().isGroup() && nSubGroups <= 1) {
			} else {
				handleField(field, reader);
			}
		}
		
		KJ_IF_MAYBE(pField, reader.which()) {
			auto& field = *pField;
			if(field.getProto().isGroup() && nSubGroups <= 1) {
				nodeCaption = kj::strTree(mv(nodeCaption), "\n", "-- ", field.getProto().getName(), " --");
				handleGroup(reader.get(field).as<capnp::DynamicStruct>());
			} else {
			}
		}
				
		for(auto field : schema.getNonUnionFields()) {
			if(field.getProto().isGroup() && nSubGroups <= 1) {
				nodeCaption = kj::strTree(mv(nodeCaption), "\n", "-- ", field.getProto().getName(), " --");
				handleGroup(reader.get(field).as<capnp::DynamicStruct>());
			} else {
			}
		}
	};
	
	handleGroup(reader);
	
	addGraphNode(graph, nodeId, nodeCaption.flatten());
	return nodeId;
}

uint64_t addInterfaceToGraph(py::object graph, capnp::DynamicCapability::Client client) {
	auto schema = client.getSchema();
	
	uint64_t nodeId = nodeUUID();
	addGraphNode(graph, nodeId, kj::str(schema));
	
	return nodeId;
}
	
kj::Vector<uint64_t> addDynamicToGraph(py::object graph, capnp::DynamicValue::Reader reader) {
	auto type = reader.getType();
		
	kj::Vector<uint64_t> result;
	if(type == capnp::DynamicValue::STRUCT) {
		result.add(addStructToGraph(graph, reader.as<capnp::DynamicStruct>()));
		
	} else if(type == capnp::DynamicValue::CAPABILITY) {
		result.add(addInterfaceToGraph(graph, reader.as<capnp::DynamicCapability>()));
		
	} else if(type == capnp::DynamicValue::LIST) {
		auto asList = reader.as<capnp::DynamicList>();
		auto wrapped = asList.getSchema().getElementType();
		
		for(auto entry : asList) {
			if(wrapped.isList()) {
				if(canAddToGraph(wrapped.asList().getElementType())) {
					uint64_t nodeId = nodeUUID();
					addGraphNode(graph, nodeId, "List");
					
					for(uint64_t child : addDynamicToGraph(graph, entry)) {
						addGraphEdge(graph, nodeId, child, "");
					}
					
					result.add(nodeId);
				} else {
					uint64_t nodeId = nodeUUID();
					addGraphNode(graph, nodeId, kj::str(
						"List\n",
						entry
					));
					result.add(nodeId);
				}
			} else if(wrapped.isStruct()) {
				result.add(addStructToGraph(graph, entry.as<capnp::DynamicStruct>()));
			} else if(wrapped.isInterface()) {
				result.add(addInterfaceToGraph(graph, entry.as<capnp::DynamicCapability>()));
			} else {
				KJ_FAIL_REQUIRE("Added un-renderable list type to graph");
			}
		}
	} else {
		KJ_FAIL_REQUIRE("Added un-renderable list type to graph", type);
	}
	
	return result;
}

}

py::object visualizeGraph(capnp::DynamicStruct::Reader reader, py::kwargs kwargs) {
	py::object graph = py::module_::import("graphviz").attr("Digraph")(**kwargs);
	addStructToGraph(graph, reader);
	
	return graph;
}

}