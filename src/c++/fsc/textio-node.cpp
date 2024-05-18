// Methods for saving and loading into textio::Node

#include "textio.h"

namespace fsc { namespace textio {

namespace {
	struct NodeStack : public Visitor {
		struct WriteInto {};
		struct Append {};
		struct WriteObject {};
		struct Done {};
		using Field = Node;
		
		using State = OneOf<WriteInto, WriteObject, Append, Field, Done>;
		
		kj::Vector<Node*> stack;
		State state = WriteInto();
		
		NodeStack(Node& first) {
			stack.add(&first);
			this -> supportsIntegerKeys = true;
		}
		
		Node& back() {
			KJ_ASSERT(!stack.empty());
			return *stack.back();
		}
		
		void push(Node& n) {
			stack.add(&n);
		}
		
		void pop() {
			stack.removeLast();
			
			if(stack.empty()) {
				state.init<Done>();
				return;
			}
			
			if(back().payload.is<Node::MapPayload>()) {
				state.init<WriteObject>();
				return;
			}
			
			if(back().payload.is<Node::ListPayload>()) {
				state.init<Append>();
				return;
			}
			
			KJ_FAIL_REQUIRE("Failed to restore state");
		}
		
		void prepareTop() {
			KJ_REQUIRE(!state.is<WriteObject>(), "Can not write sub-object without field or field name");
			KJ_REQUIRE(!state.is<Done>());
			
			if(state.is<WriteInto>()) {
				return;
			}
			
			if(state.is<Append>()) {
				Node& dst = back().payload.get<Node::ListPayload>().add();
				push(dst);
				return;
			}
			
			if(state.is<Field>()) {
				Tuple<Node, Node>& container = back().payload.get<Node::MapPayload>().add(kj::tuple(
					mv(state.get<Field>()), Node()
				));
				
				push(kj::get<1>(container));
				return;
			}
			
			state.allHandled<5>();
		}
		
		void beginObject(Maybe<size_t> size) override {
			prepareTop();
			
			back().payload.init<Node::MapPayload>();
			state.init<WriteObject>();
			
			return;
		}
		
		void endObject() override {
			KJ_REQUIRE(!state.is<Done>());
			pop();
		}
		
		void beginArray(Maybe<size_t> size) override {
			prepareTop();
			
			back().payload.init<Node::ListPayload>();
			
			KJ_IF_MAYBE(pSize, size) {
				back().payload.get<Node::ListPayload>().reserve(*pSize);
			}
			
			state.init<Append>();
			
			return;
		}
		
		void endArray() override {
			KJ_REQUIRE(!state.is<Done>());
			pop();
		}
		
		bool done() override {
			return state.is<Done>();
		}
		
		template<typename T>
		void acceptValue(T t) {
			if(state.is<WriteObject>()) {
				Node& s = state.init<Field>();
				s.payload = mv(t);
			} else {
				prepareTop();
				back().payload = mv(t);
				pop();
			}
		}
		
		void acceptNull() override {
			acceptValue(Node::NullValue());
		}
		
		void acceptDouble(double v) override {
			acceptValue(v);
		}
		
		void acceptInt(int64_t v) override {
			acceptValue(v);
		}
		
		void acceptUInt(uint64_t v) override {
			acceptValue(v);
		}
		
		void acceptBool(bool v) override {
			acceptValue(v);
		}
		
		void acceptString(kj::StringPtr v) override {
			acceptValue(kj::heapString(v));
		}
		
		void acceptData(kj::ArrayPtr<const byte> v) override {
			acceptValue(kj::heapArray<byte>(v));
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
		v.acceptDouble(p.get<double>());
	
	if(p.is<uint64_t>())
		v.acceptUInt(p.get<uint64_t>());
	
	if(p.is<int64_t>())
		v.acceptInt(p.get<int64_t>());
	
	if(p.is<bool>())
		v.acceptInt(p.get<bool>());
	
	if(p.is<kj::String>())
		v.acceptString(p.get<kj::String>());
	
	if(p.is<kj::Array<kj::byte>>())
		v.acceptData(p.get<kj::Array<kj::byte>>());
	
	if(p.is<Node::ListPayload>()) {
		auto& lpl = p.get<Node::ListPayload>();
		
		v.beginArray(lpl.size());
		for(Node& el : lpl)
			save(el, v);
		v.endArray();
	}
	
	if(p.is<Node::MapPayload>()) {
		auto& mpl = p.get<Node::MapPayload>();
		
		v.beginObject(mpl.size());
		for(auto& row : mpl) {
			save(kj::get<0>(row), v);
			save(kj::get<1>(row), v);
		}
		v.endObject();
	}
}

void save(Node&& n, Visitor& v) {
	auto& p = n.payload;
	
	if(p.is<Node::ListPayload>()) {
		auto& lpl = p.get<Node::ListPayload>();
		
		v.beginArray(lpl.size());
		for(Node& el : lpl)
			save(mv(el), v);
		v.endArray();
	} else if(p.is<Node::MapPayload>()) {
		auto& mpl = p.get<Node::MapPayload>();
		
		v.beginObject(mpl.size());
		for(auto& row : mpl) {
			save(mv(kj::get<0>(row)), v);
			save(mv(kj::get<1>(row)), v);
		}
		v.endObject();
	} else {
		save((Node&) n, v);
	}
	
	// Clear pointer nodes after saving
	// (don't clear non-pointer nodes, no point since
	// that doesn't shrink the node size)
	switch(n.payload.which()) {
		case Node::Payload::tagFor<Node::ListPayload>():
		case Node::Payload::tagFor<Node::MapPayload>():
		case Node::Payload::tagFor<kj::String>():
		case Node::Payload::tagFor<kj::Array<kj::byte>>():
			n.payload.init<Node::NullValue>();
			break;
		default:
			break;
	}
}

}}