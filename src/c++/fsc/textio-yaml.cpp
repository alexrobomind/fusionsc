#include <yaml-cpp/yaml.h>

#include <kj/encoding.h>

namespace fsc { namespace textio {

namespace {
	struct YAMLVisitor : public Visitor {
		YAML::Emitter& emitter;
		bool listStart = false;
		
		void doListBegin(bool flow) {
			if(listStart) {
				if(flow)
					emitter << YAML::Flow;
				emitter << YAML::BeginSeq;
				listStart = false;
			}
		}
		
		void beginObject(Maybe<size_t>) override {
			doListBegin(false);
			emitter << YAML::BeginMap;
		}
		
		void endObject() override {
			doListBegin(false); // In case API is misused
			emitter << YAML::EndMap;
		}
		
		void beginArray(Maybe<size_t>) override {
			doListBegin(false);
			listStart = true;
		}
		
		void endArray(Maybe<size_t>) override {
			// Perhaps the list was empty
			doListBegin(true);
			emitter << YAML::EndSeq;
		}
		
		void acceptNull() override {
			// Null in list should in principle not happen, but who knows...
			// Let's use block style to be sure
			doListBegin(false);
			emitter << YAML::Null;
		}
		
		void acceptBool(bool v) override {
			doListBegin(true);
			emitter << v;
		}
		
		void acceptDouble(double v) override {
			doListBegin(true);
			emitter << v;
		}
		
		void acceptInt(int64_t v) override {
			doListBegin(true);
			emitter << v;
		}
		
		void acceptUInt(uint64_t v) override {
			doListBegin(true);
			emitter << v;
		}
		
		void acceptString(kj::StringPtr v) override {
			doListBegin(false);
			emitter << v.cStr();
		}
		
		void acceptData(kj::ArrayPtr<const kj::byte> v) override {
			doListBegin(false);
			emitter << YAML::Binary(v.begin(), v.size());
		}

		void acceptKey(kj::StringPtr key) override {
			doListBegin(false);
			emitter << YAML::Key << key.cStr() << YAML::Value;
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
		
		State& state() { return states.back(); }
		
		std::list<Tape> tapes;
		kj::TreeMap<Anchor, Node> anchors;
		
		#define ACCEPT_IMPL_NOFIN(callExpr) \
			for(auto& t : tapes) { \
				t -> visitor -> callExpr; \
			} \
			visitor -> callExpr;
		
		#define ACCEPT_IMPL(callExpr) \
			ACCEPT_IMPL_NOFIN(callExpr) \
			checkTapes();
		
		void checkTapes() {
			for(auto it = tapes.begin(); it != tapes.end();) {
				auto& t = *it;
				if(t -> visitor -> done()) {
					auto row = anchors.insert(t -> anchor, mv(t -> node));					
					(it++) -> erase();
				} else {
					++it;
				}
			}
		}
		
		void checkAnchor(Anchor a) {
			auto ownedName = mv(anchorName);
			
			if(a == NULL_ANCHOR)
				return;
			
			tapes.add(a, ownedName);
		}	
		
		void OnDocumentStart(const Mark&) override {}
		void OnDocumentEnd() override {}
		
		void OnNull(Mark& mark, Anchor a) override {
			KJ_REQUIRE(state() == VALUE || State() == MAP_VALUE, "Null values only supported as values");
			
			checkAnchor(a);
			ACCEPT_IMPL(acceptNull());
		}
		
		void OnAlias(Mark& mark, Anchor a) override {
			Node& node = aliases.get(a);
			
			switch(state()) {
				case MERGE_KEY:
					state() = MAP_KEY;
					// No break
					
				case MERGE_KEY_ENTRY: {
					kj::StringPtr asKJ = value;
					
					// Retrieve target
					Node& node = anchors.get(anchor);
					
					KJ_REQUIRE(node.payload.is<Node::MapPayload>(), "Can only merge map-type payloads");
					for(auto& row : node.payload.get<Node::MapPayload>()) {
						ACCEPT_IMPL_NOFIN(acceptKey(row.key))
						
						for(auto& tape : tapes)
							save(row.value, tape -> visitor);
						save(row.value, visitor);
					}
				}
				
				case MAP_VALUE:
					state() = MAP_ENTRY;
					// No break
				
				case VALUE: {
					for(auto& tape : tapes)
						save(node, tape -> visitor);
					save(node, visitor);
					checkTapes();
					break;
				}
				
				case MAP_KEY:
					KJ_FAIL_REQUIRE("API misuse");
			}
		}
		
		void OnScalar(Mark& mark, const std::string& tag, Anchor anchor, const std::string& value) override {
			switch(state()) {
				case MAP_VALUE:
					state() = MAP_KEY;
					// No break
					
				case VALUE: {
					checkAnchor(anchor);
					
					kj::StringPtr asKJ = value;
					if(tag == "!") {
						ACCEPT_IMPL(acceptString(asKJ))
						break;
					}
					
					if(tag == "!!binary") {
						auto decResult = kj::decodeBase64(asKJ);
						ACCEPT_IMPL(acceptData(decResult))
						break;
					}
					
					KJ_IF_MAYBE(pUInt, asKJ.tryParseAs<uint64_t>()) {
						ACCEPT_IMPL(acceptUInt(*pUInt))
						break;
					}
					
					KJ_IF_MAYBE(pInt, asKJ.tryParseAs<int64_t>()) {
						ACCEPT_IMPL(acceptInt(*pInt))
						break;
					}
					
					KJ_IF_MAYBE(pFloat, asKJ.tryParseAs<double>()) {
						ACCEPT_IMPL(acceptDouble(*pFloat))
						break;
					}
					
					ACCEPT_IMPL(acceptString(asKJ))
					break;
				}
				
				case MAP_KEY: {
					kj::StringPtr asKJ = value;
					
					if(tag == "?" && asKJ == "<<") {
						state() = MERGE_KEY;
						break;
					}
					
					ACCEPT_IMPL_NOFIN(acceptKey(asKJ))
					state() = MAP_VALUE;
					
					break;
				}
				
				case MERGE_KEY:
				case MERGE_KEY_ENTRY:
					KJ_FAIL_REQUIRE("API Misuse");
			}
		}
		
		void OnSequenceStart(const Mark&, const std::string& tag, Anchor anchor, YAML::EmitterStyle::value) override {
			checkAnchor(anchor);
			
			switch(state()) {
				case VALUE:
				case MAP_VALUE: 
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


}}