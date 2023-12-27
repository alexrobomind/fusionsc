// Methods for saving and loading into textio::Node

namespace fsc { namespace textio {

namespace {
	struct NodeStack : public Visitor {
		struct WriteInto {};
		struct Append {};
		struct WriteObject {};
		struct Done {};
		using Field = capnp::String;
		
		using State = OneOf<WriteInto, WriteObject, Append, Field, Done>;
		
		kj::Vector<Node&> stack;
		State state = WriteInto();
		
		NodeStack(Node& first) {
			stack.append(first);
		}
		
		Node& back() {
			KJ_ASSERT(!stack.empty());
			return stack.back();
		}
		
		void push(Node& n) {
			stack.add(n);
		}
		
		void pop() {
			stack.removeLast();
			
			if(stack.empty()) {
				state.emplace<Done>();
				return;
			}
			
			if(back().payload.is<Node::MapPayload>()) {
				state.emplace<WriteObject>();
				return;
			}
			
			if(back().payload.is<Node::ListPayload>()) {
				state.emplace<Append>();
				return;
			}
			
			KJ_FAIL_REQUIRE("Failed to restore state");
		}
		
		void prepareTop() {
			KJ_REQUIRE(!state.is<WriteObject>(), "Can not write sub-object without field or field name");
			KJ_REQUIRE(!state.is<Done>());
			
			if(state.is<WriteInto>()) {
				dst = &top;
				dstType = type();
				return;
			}
			
			if(state.is<Append>()) {
				Node& dst = top.payload.as<Node::ListPayload>().add();
				dstType = elementType(type());
				push(dst, dstType);
				return;
			}
			
			if(state.is<Field>()) {
				Node& dst = &(top.payload.as<Node::MapPayload>().insert(state.get<Field>()));
				auto dstType = fieldType(top.type);
				push(dst, dstType);
				return;
			}
			
			state.allHandled<5>();
		}
		
		void beginObject(Maybe<size_t> size) override {
			prepareTop();
			
			checkStruct(type());
			back().payload.emplace<Node::MapPayload>();
			state.emplace<WriteObject>();
			
			return;
		}
		
		void endObject() override {
			pop();
		}
		
		void beginArray(Maybe<size_t> size) override {
			prepareTop();
			
			checkList(type());
			back().payload.emplace<Node::ListPayload>();
			
			KJ_IF_MAYBE(pSize, size) {
				back().payload.as<Node::ListPayload>().reserve(*pSize);
			}
			
			state.emplace<Append>();
			
			return;
		}
		
		void endArray() override {
			pop();
		}
		
		bool done() override {
			return state.is<Done>();
		}
		
		void acceptKey(kj::StringPtr key) override {
			KJ_REQUIRE(state.is<WriteObject>());
			state = kj::heapString(key);
		}
		
		void acceptNull() override {
			prepareTop();
			back().payload.emplace<capnp::Void>();
			pop();
		}
		
		void acceptDouble(double v) override {
			prepareTop();
			back().payload = v;
			pop();
		}
		
		void acceptInt(int64_t v) override {
			prepareTop();
			back().payload = v;
			pop();
		}
		
		void acceptUInt(uint64_t v) override {
			prepareTop();
			back().payload = v;
			pop();
		}
		
		void acceptBool(bool v) override {
			prepareTop();
			back().payload = v;
			pop();
		}
		
		void acceptString(kj::StringPtr v) override {
			prepareTop();
			back().payload = kj::heapString(v);
			pop();
		}
		
		void acceptData(kj::ArrayPtr<const byte> v) override {
			prepareTop();
			back().payload = kj::heapArray<byte>(v);
			pop();
		}
	};
}

Own<Visitor> createVisitor(Node& n) {
	return kj::heap<NodeStack>(n);
}

void save(Node& n, Visitor& v) {
	auto& p = n.payload;
	
	if(p.is<Node::NullValue>())
		v.acceptNull();
	
	if(p.is<double>())
		v.acceptDouble(p.as<double>());
	
	if(p.is<uint64_t>())
		v.acceptUInt(p.as<uint64_t>());
	
	if(p.is<int64_t>())
		v.acceptInt(p.as<int64_t>());
	
	if(p.is<bool>())
		v.acceptInt(p.as<bool>());
	
	if(p.is<kj::String>())
		v.acceptString(p.as<kj::String>());
	
	if(p.is<kj::Array<kj::byte>())
		v.acceptData(p.as<kj::Array<kj::byte>>());
	
	if(p.is<Node::ListPayload>()) {
		auto& lpl = p.as<Node::ListPayload>();
		
		v.beginArray(lpl.size());
		for(Node& el : lpl)
			save(el, v);
		v.endArray();
	}
	
	if(p.is<Node::MapPayload>()) {
		auto& mpl = p.as<Node::MapPayload>();
		
		v.beginObject();
		for(auto& row : mpl) {
			v.acceptKey(row.key);
			save(row.value, v);
		}
		v.endObject();
	}
}

}}