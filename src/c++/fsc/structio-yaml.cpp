#include "structio.h"

#include <yaml-cpp/yaml.h>
#include <yaml-cpp/eventhandler.h>
#include <kj/encoding.h>

namespace fsc { namespace structio {

namespace {
	struct YAMLVisitor : public Visitor {
		YAML::Emitter& emitter;
		
		enum State {
			VALUE, LIST_START, MAP_KEY, MAP_VALUE, DONE
		};
		kj::Vector<State> states;
		
		YAMLVisitor(YAML::Emitter& e) :
			emitter(e)
		{
			this -> supportsIntegerKeys = true;
			states.add(VALUE);
		}
		
		State& state() { return states.back(); }
				
		void beginValue(bool flow) {
			KJ_REQUIRE(state() != DONE, "API misuse");
			
			if(state() == LIST_START) {
				if(flow)
					emitter << YAML::Flow;
				emitter << YAML::BeginSeq;
				state() = VALUE;
			}
			if(state() == MAP_KEY) {
				emitter << YAML::Key;
				state() = MAP_VALUE;
			}
			if(state() == MAP_VALUE) {
				emitter << YAML::Value;
				state() = MAP_KEY;
			}
		}
		
		void push(State s) {
			states.add(s);
		}
		
		void pop() {
			states.removeLast();
			KJ_REQUIRE(!states.empty(), "API misuse");
			
			if(states.size() == 1)
				state() = DONE;
		}
				
		void beginObject(Maybe<size_t>) override {
			beginValue(false);
			push(MAP_KEY);
			
			emitter << YAML::BeginMap;
		}
		
		void endObject() override {
			KJ_REQUIRE(state() == MAP_KEY, "API misuse");
			pop();
			
			emitter << YAML::EndMap;
		}
		
		void beginArray(Maybe<size_t>) override {
			beginValue(false);
			push(LIST_START);
			// Emission follows later
		}
		
		void endArray() override {
			// Perhaps the list was empty
			beginValue(true);
			pop();
			
			emitter << YAML::EndSeq;
		}
		
		void acceptNull() override {
			// Null in list should in principle not happen, but who knows...
			// Let's use block style to be sure
			beginValue(false);
			emitter << YAML::Null;
			
			if(states.size() == 1)
				state() = DONE;
		}
		
		void acceptBool(bool v) override {
			beginValue(true);
			emitter << v;
			
			if(states.size() == 1)
				state() = DONE;
		}
		
		void acceptDouble(double v) override {
			beginValue(true);
			emitter << v;
			
			if(states.size() == 1)
				state() = DONE;
		}
		
		void acceptInt(int64_t v) override {
			beginValue(true);
			emitter << v;
			
			if(states.size() == 1)
				state() = DONE;
		}
		
		void acceptUInt(uint64_t v) override {
			beginValue(true);
			emitter << v;
			
			if(states.size() == 1)
				state() = DONE;
		}
		
		void acceptString(kj::StringPtr v) override {
			beginValue(false);
			emitter << v.cStr();
			
			if(states.size() == 1)
				state() = DONE;
		}
		
		void acceptData(kj::ArrayPtr<const kj::byte> v) override {
			beginValue(false);
			emitter << YAML::Binary(v.begin(), v.size());
			
			if(states.size() == 1)
				state() = DONE;
		}
		
		bool done() override {
			return state() == DONE;
		}
	};
	
	struct YAMLEventHandler : public YAML::EventHandler {
		using Mark = YAML::Mark;
		using Anchor = YAML::anchor_t;
		
		static constexpr Anchor NULL_ANCHOR = YAML::NullAnchor;
		
		struct Tape {
			Node node;
			Own<Visitor> visitor;
			Anchor anchor;
			
			Tape(Anchor a) : visitor(createVisitor(node)), anchor(a) {};
			Tape(Tape&) = delete;
			Tape(Tape&&) = delete;
		};
		
		enum State {
			VALUE, MAP_KEY, MAP_VALUE, MERGE_KEY, MERGE_KEY_ENTRY
		};
		
		Visitor& visitor;
		kj::Vector<State> states;
		
		std::list<Tape> tapes;
		kj::TreeMap<Anchor, Node> anchors;
		
		YAMLEventHandler(Visitor& v) :
			visitor(v)
		{
			states.add(VALUE);
		}
		
		~YAMLEventHandler() noexcept {}
		
		State& state() { return states.back(); }
		
		#define ACCEPT_IMPL_NOFIN(callExpr) \
			for(auto& t : tapes) { \
				t.visitor -> callExpr; \
			} \
			visitor.callExpr;
		
		#define ACCEPT_IMPL(callExpr) \
			ACCEPT_IMPL_NOFIN(callExpr) \
			checkTapes();
		
		void checkTapes() {
			for(auto it = tapes.begin(); it != tapes.end();) {
				auto& t = *it;
				if(t.visitor -> done()) {
					anchors.insert(t.anchor, mv(t.node));					
					tapes.erase(it++);
				} else {
					++it;
				}
			}
		}
		
		void checkAnchor(Anchor a) {			
			if(a == NULL_ANCHOR)
				return;
			
			tapes.emplace(tapes.end(), a);
		}	
		
		void OnDocumentStart(const Mark&) override {}
		void OnDocumentEnd() override {}
		
		void OnNull(const Mark& mark, Anchor a) override {
			switch(state()) {
				case MERGE_KEY:
				case MERGE_KEY_ENTRY:
					KJ_FAIL_REQUIRE("Merge key must be alias");
					break;
				
				case MAP_VALUE:
					state() = MAP_KEY;
					// No break
				case VALUE:
					checkAnchor(a);
					ACCEPT_IMPL(acceptNull());
					break;
				
				case MAP_KEY:
					KJ_FAIL_REQUIRE("Null map keys not supported");
			}
		}
		
		void OnAlias(const Mark& mark, Anchor a) override {
			Node* nodePtr = nullptr;
			KJ_IF_MAYBE(pNode, anchors.find(a)) {
				nodePtr = pNode;
			}
			Node& node = *nodePtr;
			
			switch(state()) {
				case MERGE_KEY:
					state() = MAP_KEY;
					// No break
					
				case MERGE_KEY_ENTRY: {										
					KJ_REQUIRE(node.payload.is<Node::MapPayload>(), "Can only merge map-type payloads");
					for(auto& row : node.payload.get<Node::MapPayload>()) {
						for(auto& tape : tapes) {
							save(kj::get<0>(row), *tape.visitor);
							save(kj::get<1>(row), *tape.visitor);
						}
						
						save(kj::get<0>(row), visitor);
						save(kj::get<1>(row), visitor);
					}
					break;
				}
				
				case MAP_VALUE:
					state() = MAP_KEY;
					// No break
				
				case VALUE: {
					for(auto& tape : tapes)
						save(node, *tape.visitor);
					save(node, visitor);
					checkTapes();
					break;
				}
				
				case MAP_KEY:
					KJ_FAIL_REQUIRE("API misuse");
			}
		}
		
		void handleScalar(const std::string& value, const std::string& tag) {
			kj::StringPtr asKJ = value;
			if(tag == "!") {
				ACCEPT_IMPL(acceptString(asKJ))
				return;
			}
			
			if(tag == "!!binary") {
				auto decResult = kj::decodeBase64(asKJ);
				ACCEPT_IMPL(acceptData(decResult))
				return;
			}
			
			KJ_IF_MAYBE(pUInt, asKJ.tryParseAs<uint64_t>()) {
				ACCEPT_IMPL(acceptUInt(*pUInt))
				return;
			}
			
			KJ_IF_MAYBE(pInt, asKJ.tryParseAs<int64_t>()) {
				ACCEPT_IMPL(acceptInt(*pInt))
				return;
			}
			
			KJ_IF_MAYBE(pFloat, asKJ.tryParseAs<double>()) {
				ACCEPT_IMPL(acceptDouble(*pFloat))
				return;
			}
			
			if(asKJ.startsWith(".inf") || asKJ.startsWith(".Inf")) {
				ACCEPT_IMPL(acceptDouble(std::numeric_limits<double>::infinity()));
				return;
			}
			
			if(asKJ.startsWith("-.inf") || asKJ.startsWith("-.Inf")) {
				ACCEPT_IMPL(acceptDouble(-std::numeric_limits<double>::infinity()));
				return;
			}
			
			if(asKJ.startsWith(".nan") || asKJ.startsWith(".NaN")) {
				ACCEPT_IMPL(acceptDouble(std::numeric_limits<double>::quiet_NaN()));
				return;
			}
			
			ACCEPT_IMPL(acceptString(asKJ))
		}
		
		void OnScalar(const Mark& mark, const std::string& tag, Anchor anchor, const std::string& value) override {
			switch(state()) {
				case MAP_VALUE:
					checkAnchor(anchor);
					handleScalar(value, tag);
					state() = MAP_KEY;
					break;
									
				case VALUE:
					checkAnchor(anchor);
					handleScalar(value, tag);
					break;
				
				case MAP_KEY: {
					kj::StringPtr asKJ = value;
					
					if(tag == "?" && asKJ == "<<") {
						state() = MERGE_KEY;
						break;
					}
					
					handleScalar(value, tag);
					state() = MAP_VALUE;
					
					break;
				}
				
				case MERGE_KEY:
				case MERGE_KEY_ENTRY:
					KJ_FAIL_REQUIRE("Merge keys must be aliases");
			}
		}
		
		void OnSequenceStart(const Mark&, const std::string& tag, Anchor anchor, YAML::EmitterStyle::value) override {
			checkAnchor(anchor);
			
			switch(state()) {
				case MAP_VALUE: 
					state() = MAP_KEY;
					// No break
				case VALUE:
					states.add(VALUE);
					ACCEPT_IMPL_NOFIN(beginArray(nullptr))
					break;
				
				case MERGE_KEY:
					state() = MERGE_KEY_ENTRY;
					break;
				
				case MAP_KEY:
				case MERGE_KEY_ENTRY:
					KJ_FAIL_REQUIRE("API misuse");
			}
		}
		
		void OnSequenceEnd() override {
			switch(state()) {
				case VALUE:
					ACCEPT_IMPL(endArray())
					
					states.removeLast();
					if(state() == MAP_VALUE)
						state() = MAP_KEY;
					
					break;
				
				case MERGE_KEY_ENTRY:
					state() = MAP_KEY;
					break;
					
				case MAP_VALUE:
				case MERGE_KEY:				
				case MAP_KEY:
					KJ_FAIL_REQUIRE("API misuse");
			}
		}
		
		void OnMapStart(const Mark&, const std::string& tag, Anchor anchor, YAML::EmitterStyle::value) override {
			checkAnchor(anchor);
			
			switch(state()) {
				case MAP_VALUE:
					state() = MAP_KEY;
					// No break
				case VALUE:
					ACCEPT_IMPL(beginObject(nullptr))
					states.add(MAP_KEY);
					break;
				
				case MERGE_KEY:
				case MERGE_KEY_ENTRY:
				case MAP_KEY:
					KJ_FAIL_REQUIRE("API misuse");
			}
		}
		
		void OnMapEnd() override {
			switch(state()) {
				case MAP_KEY:
					ACCEPT_IMPL(endObject())
					states.removeLast();
					break;
				
				case VALUE:
				case MAP_VALUE:
				case MERGE_KEY:
				case MERGE_KEY_ENTRY:
					KJ_FAIL_REQUIRE("API misuse");
			}
		}
	};
}

Own<Visitor> createVisitor(YAML::Emitter& e) {
	return kj::heap<YAMLVisitor>(e);
}

namespace internal {
	void yamlcppLoad(kj::BufferedInputStream& bis, Visitor& v, const Dialect&) {
		auto inStream = asStdStream(bis);
		YAML::Parser parser(*inStream);
		
		YAMLEventHandler handler(v);
		parser.HandleNextDocument(handler);
	}
	Own<Visitor> createYamlcppWriter(kj::BufferedOutputStream& bos, const Dialect& d) {
		auto outStream = asStdStream(bos);
		auto emitter = kj::heap<YAML::Emitter>(*outStream);
		auto visitor = kj::heap<YAMLVisitor>(*emitter);
		
		return visitor.attach(mv(outStream), mv(emitter));
	}
}

}}

YAML::Emitter& operator<<(YAML::Emitter& e, capnp::DynamicValue::Reader r) {
	fsc::structio::SaveOptions so;
	so.compact = true;
	
	fsc::structio::save(r, *fsc::structio::createVisitor(e), so);
	
	return e;
}