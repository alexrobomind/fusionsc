"""
Helper module to build representation graphs for data visualization
"""

from . import capnp

from typing import List

class GraphBuilder:
	_uuidCounter: int = 0
	graph: "graphviz.Digraph"
	
	def __init__(self, graph = None):
		import graphviz
		
		if graph is None:
			self.graph = graphviz.Digraph()
		else:
			self.graph = graph
	
	def _uuiud(self) -> str:
		tmp = self._uuidCounter
		self._uuidCounter += 1
		
		return str(tmp)
	
	def _shouldSeparate(self, t) -> bool:		
		if isinstance(t, capnp.ListSchema):
			t = t.elementType
		
		return isinstance(t, (capnp.ListSchema, capnp.StructSchema, capnp.InterfaceSchema))
	
	def _addStruct(self, reader) -> str:
		schema = reader.type_
		caption = str(schema)
		
		nodeId = self._uuiud()
		
		caption = str(schema)
		
		def handleField(reader, field):
			nonlocal caption
			
			fieldType = field.type
			fieldName = str(field.proto.name)
			
			if not reader.has_(fieldName):
				return
			
			val = reader[fieldName]
			
			if self._shouldSeparate(field.type):
				children = self.addMany(val)
				
				for child in children:
					self.graph.edge(nodeId, child, fieldName)
			else:
				caption += f"\n{fieldName} = {val}"
		
		def handleGroup(group):
			nonlocal caption
			
			groupSchema = group.type_
			groupFields = groupSchema.fields
			
			activeFields = groupSchema.nonUnionFields
			w = group.which_()
			if w:
				activeFields[w] = groupSchema.unionFields[w]
			
			# Count subgroups
			nGroups = 0
			
			def isGroup(f):
				if f is None:
					return False
				
				return f.proto.which_() == "group"
			
			groupFields = [name for name, f in activeFields.items() if isGroup(f)]
			
			# Show all non-group fields
			for f in activeFields.values():
				if isGroup(f):
					continue
				
				handleField(group, f)
			
			if len(groupFields) <= 1:
				# If only 1 group, just append its content
				for f in groupFields:
					caption += f"\n-- {f} --"
					handleGroup(group[f])
			else:
				for f in groupFields:
					handleField(group[f])
		
		handleGroup(reader)
		
		self.graph.node(nodeId, caption)
		return nodeId
	
	def _addInterface(self, client) -> str:
		nodeId = self._uuiud()
		self.graph.node(nodeId, str(client.type_))
		return nodeId
		
	def addOne(self, val) -> str:
		while hasattr(val, "_fusionsc_wraps"):
			val = val._fusionsc_wraps
			
		t = val.type_
		
		if isinstance(t, capnp.StructSchema):
			return self._addStruct(val)
		if isinstance(t, capnp.InterfaceSchema):
			return self._addInterface(val)
			
		assert isinstance(t, capnp.ListSchema), "Can only add structs, list, and interfaces to the graph"
		elType = t.elementType
		
		nodeId = self._uuiud()
		
		# List of compound objects get a root node, then we add all the entries
		if isinstance(elType, (capnp.StructSchema, capnp.InterfaceSchema, capnp.ListSchema)):	
			self.graph.node(nodeId, "List")		
			for c in self.addMany(val):
				self.graph.edge(nodeId, c, "Item")
		else:
			self.graph.node(nodeId, "\n".join([str(item) for item in val]))
			
		return nodeId
	
	def addMany(self, val) -> List[str]:
		while hasattr(val, "_fusionsc_wraps"):
			val = val._fusionsc_wraps
			
		t = val.type_
		
		if isinstance(t, capnp.StructSchema):
			return [self._addStruct(val)]
		if isinstance(t, capnp.InterfaceSchema):
			return [self._addInterface(val)]
		
		assert isinstance(t, capnp.ListSchema), "Can only add structs, list, and interfaces to the graph"
		
		elType = t.elementType
		
		if isinstance(elType, (capnp.StructSchema, capnp.InterfaceSchema, capnp.ListSchema)):
			return [self.addOne(item) for item in val]
		
		# If the element type is something that can not be added to the graph, we need
		# to fall back to addOne which can process all kinds of lists
		return [self.addOne(val)]		
	
		
