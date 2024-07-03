#include "capfuzz.h"

#include <kj/memory.h>
#include <kj/function.h>
#include <kj/async-queue.h>
#include <capnp/capability.h>

#include <list>
#include <algorithm>

using kj::byte;
using kj::Array;
using kj::ArrayPtr;
using kj::Maybe;
using kj::Promise;
using kj::PromiseFulfiller;
using kj::Own;
using kj::mv;
using kj::cp;

using kj::READY_NOW;
using kj::NEVER_DONE;

using capnp::DynamicCapability;
using capnp::DynamicStruct;
using capnp::DynamicList;
using capnp::DynamicValue;

using capnp::StructSchema;
using capnp::InterfaceSchema;

using capnp::ClientHook;

namespace capfuzz {

namespace {

struct FuzzInput {
	FuzzInput(ArrayPtr<const byte> data) :
		data(data)
	{}
	
	bool done() {
		return offset >= data.size();
	}
	
	uint8_t readOne() {
		if(done())
			return 0;
		
		const uint8_t* ptr = data.begin() + offset;
		++offset;
		return *ptr;
	}
	
	template<typename T>
	T read() {
		T result;
		
		byte* asByteArray = reinterpret_cast<byte*>(&result);
		for(size_t i = 0; i < sizeof(T); ++i) {
			asByteArray[i] = readOne();
		}
		
		return result;
	}
	
	capnp::DynamicValue::Reader readNonPointer(capnp::Type type) {
		switch(type.which()) {
			#define HANDLE_PRIMITIVE(SchemaType, CppType) \
				case capnp::schema::Type::SchemaType: \
					return read<CppType>();
			
			HANDLE_PRIMITIVE(BOOL, bool);
			
			HANDLE_PRIMITIVE(INT8, int8_t);
			HANDLE_PRIMITIVE(INT16, int16_t);
			HANDLE_PRIMITIVE(INT32, int32_t);
			HANDLE_PRIMITIVE(INT64, int64_t);
			
			HANDLE_PRIMITIVE(UINT8, uint8_t);
			HANDLE_PRIMITIVE(UINT16, uint16_t);
			HANDLE_PRIMITIVE(UINT32, uint32_t);
			HANDLE_PRIMITIVE(UINT64, uint64_t);
			
			HANDLE_PRIMITIVE(FLOAT32, float);
			HANDLE_PRIMITIVE(FLOAT64, double);
			
			#undef HANDLE_PRIMITIVE
			
			case capnp::schema::Type::VOID:
				return capnp::Void();
			
			case capnp::schema::Type::ENUM: {
				auto schema = type.asEnum();
				auto enumerants = schema.getEnumerants();
				
				uint16_t idx = read<uint16_t>(enumerants.size());
				return capnp::DynamicEnum(enumerants[idx]);
			}
			
			default: break;
		}
		
		KJ_FAIL_REQUIRE("Pointer / unknown type requested for schema read", type.which());
	}
	
	template<typename T>
	T read(T maxVal) {
		KJ_REQUIRE(maxVal > 0);
		return read<T>() % maxVal;
	}

	template<typename T>
	typename std::list<T>::iterator selectFromList(std::list<T>& input) {
		KJ_REQUIRE(input.size() > 0);
		
		auto idx = read(input.size());
		auto it = input.begin();
		for(auto i : kj::range(0, idx)) ++it;
		
		return it;
	}
	
	void fill(ArrayPtr<byte> target) {
		size_t maxCopy = std::min(target.size(), data.size() - offset);
		
		if(maxCopy > 0) {
			memcpy(target.begin(), data.begin() + offset, maxCopy);
		}
		
		if(maxCopy < target.size()) {
			memset(target.begin() + maxCopy, 0, target.size() - maxCopy);
		}
		
		offset += maxCopy;
	}
	
private:
	ArrayPtr<const byte> data;
	size_t offset = 0;
};

namespace internal {
	void doCollect(capnp::InterfaceSchema schema, kj::Vector<InterfaceSchema::Method>& out) {
		for(auto m : schema.getMethods()) {
			for(auto& n : out) {
				if(m == n) {
					goto next;
				}
			}
		
			out.add(m);
			
			// Jump here if method is already present
			next:
			(void) 0;
		}
		
		for(auto super : schema.getSuperclasses())
			doCollect(schema, out);
	}
}

kj::Array<capnp::InterfaceSchema::Method> collectMethods(capnp::InterfaceSchema schema) {
	kj::Vector<InterfaceSchema::Method> out;
	internal::doCollect(schema, out);
	return out.releaseAsArray();
}

struct OutboundCall {
	capnp::RemotePromise<DynamicStruct> promise;
};

struct InboundCall {
	Own<PromiseFulfiller<void>> fulfiller;
	capnp::CallContext<DynamicStruct, DynamicStruct> ctx;
};

struct Import {
	DynamicCapability::Client cap;
	Array<InterfaceSchema::Method> methods;
	
	Import(DynamicCapability::Client newCap) :
		cap(newCap), methods(collectMethods(cap.getSchema()))
	{}
};

struct ExportImpl : public DynamicCapability::Server {
	struct PerMethod {
		InterfaceSchema::Method method;
		kj::ProducerConsumerQueue<InboundCall> calls;
		
		PerMethod(InterfaceSchema::Method m) : method(mv(m)) {};
	};
	kj::Array<PerMethod> perMethod;
	
	ExportImpl(capnp::InterfaceSchema is) :
		Server(is)
	{
		auto methods = collectMethods(is);
		auto pmBuilder = kj::heapArrayBuilder<PerMethod>(methods.size());
		for(auto m : methods) {
			pmBuilder.add(mv(m));
		}
		perMethod = pmBuilder.finish();
	}
	
	Promise<void> call(InterfaceSchema::Method method, capnp::CallContext<DynamicStruct, DynamicStruct> ctx) {
		for(auto& pm : perMethod) {
			if(pm.method != method)
				continue;
			
			auto paf = kj::newPromiseAndFulfiller<void>();
			InboundCall c { mv(paf.fulfiller), mv(ctx) };
			pm.calls.push(mv(c));
			
			return mv(paf.promise);
		}
		
		KJ_UNIMPLEMENTED("Unimplemented");
	}
	
	Promise<InboundCall> awaitCall(uint64_t method) {
		KJ_REQUIRE(perMethod.size() >= 1, "Can not await call on interfaces without methods");
		
		method %= perMethod.size();
		return perMethod[method].calls.pop();
	}
};

struct Export {
	Export(InterfaceSchema is) {
		auto impl = kj::heap<ExportImpl>(mv(is));
		pImpl = impl.get();
		cap = mv(impl);
	}
	
	Export(Export&) = default;
	Export(Export&&) = default;
	
	ExportImpl* pImpl;
	
	capnp::DynamicCapability::Client get() { return cap; }
private:
	// Keep-alive
	capnp::DynamicCapability::Client cap;
};

enum ProtocolOp {
	OP_CALL,		// Create outbound call
	OP_DELETE,		// Drop capability reference
	
	OP_CANCEL,		// Cancel outbound call
	OP_AWAIT,		// Await outbound call
	OP_PIPE,		// Create pipeline on outbound call (stores intermediate pipeline if results in struct)
	
	OP_EXTEND_PIPE, // Create pipeline on intermediate pipeline
	
	OP_ACCEPT,		// Wait for incoming call to happen on exported cap
	
	OP_FULFILL,		// Complete incoming call
	OP_REJECT,		// Reject incoming call
	
	N_OPS
};

constexpr uint8_t UNTYPED_CAPS = 5;

struct ProtocolState : public InputBuilder::Context {
	std::list<Import> imports;
	std::list<Export> exports;
	std::list<InboundCall> inboundCalls;
	std::list<OutboundCall> outboundCalls;
	
	FuzzInput input;
	ProtocolConfig config;
	
	Maybe<DynamicStruct::Pipeline> activePipeline = nullptr;
	
	ProtocolState(kj::ArrayPtr<const byte> inputData, ProtocolConfig config) :
		input(inputData), config(config)
	{}
	
	~ProtocolState() noexcept {}
	
	void addImport(DynamicCapability::Client clt) {
		auto hook = ClientHook::from(cp(clt));
		
		if(hook -> isNull() || hook -> isError())
			return;
		
		Import i { cp(clt) };
		imports.push_back(mv(i));
	}
	
	Promise<void> step() {
		uint8_t opId = input.read(static_cast<uint8_t>(N_OPS));
		
		switch(static_cast<ProtocolOp>(opId)) {
			case OP_CALL: return doCall();
			case OP_DELETE: return doDelete();
			
			case OP_CANCEL: return doCancel();
			case OP_AWAIT: return doAwait();
			case OP_PIPE: return doPipe();
			
			case OP_EXTEND_PIPE: return doExtendPipe();
			
			case OP_ACCEPT: return doAccept();
			
			case OP_FULFILL: return doFulfill();
			case OP_REJECT: return doReject();
			
			case N_OPS: break;
		}
		
		KJ_FAIL_REQUIRE("Dead code encountered");
	}
	
	Promise<void> run() {
		if(input.done())
			return READY_NOW;
		
		return step().then([this]() { return run(); });
	}

private:
	kj::Array<InputBuilder*> getBuilders(capnp::Type type) {
		kj::Vector<InputBuilder*> result;
		
		for(auto builder : config.builders) {
			for(auto i : kj::range(0, builder -> getWeight(type)))
				result.add(builder);
		}
		
		return result.releaseAsArray();
	}
	capnp::Capability::Client makeUntyped(uint8_t type) {
		#define FROM_EXCEPTION(ExcType) capnp::Capability::Client(KJ_EXCEPTION(ExcType, "Error-client"))
		switch(type) {
			case 0: return nullptr;
			case 1: return FROM_EXCEPTION(FAILED);
			case 2: return FROM_EXCEPTION(OVERLOADED);
			case 3: return FROM_EXCEPTION(DISCONNECTED);
			case 4: return FROM_EXCEPTION(UNIMPLEMENTED);
		}
		#undef FROM_EXCEPTION
		
		KJ_FAIL_REQUIRE("Invalid untyped client requested");
	}
	
	// Create a capability
	DynamicCapability::Client getCapability(InterfaceSchema is) override {
		auto generic = is.getGeneric();
		
		kj::Vector<DynamicCapability::Client> candidates;
				
		// Since we have no type inference capabilities here, we just compare
		// the unbranded schemas.
		
		auto checkCap = [&](DynamicCapability::Client cap) {
			if(cap.getSchema().getGeneric() == generic)
				candidates.add(cap);
		};
		
		for(auto& import : imports) {
			checkCap(import.cap);
		}
		for(auto& exp : exports) {
			checkCap(exp.get());
		}
		
		auto builders = getBuilders(is);
		
		size_t nCapTypes = UNTYPED_CAPS + 1 + builders.size() + candidates.size();
		size_t capType = input.read(nCapTypes);
		
		if(capType < UNTYPED_CAPS) {
			return makeUntyped((uint8_t) capType).castAs<DynamicCapability>(is);
		}
		capType -= UNTYPED_CAPS;
		
		if(capType < builders.size()) {
			auto result = builders[capType] -> getCapability(is, *this).castAs<capnp::DynamicCapability>(is);
			addImport(result);
			return result;
		}
		capType -= builders.size();
		
		if(capType == 0) {
			// Create new export
			Export x(is);
			exports.emplace_back(x);
			return x.get();
		}
		capType -= 1;
		
		return candidates[capType];
	}
	
	// Process an inbound struct (incoming call or call result)
	void handleIncoming(capnp::DynamicStruct::Reader data) {
		for(auto field : data.getSchema().getFields()) {
			auto type = field.getType();
			
			if(!data.has(field))
				continue;
			
			if(type.isStruct()) {
				handleIncoming(data.get(field).as<DynamicStruct>());
			}
			
			if(type.isInterface()) {
				addImport(data.get(field).as<DynamicCapability>());
			}
		}
	}
	
	void fillStruct(DynamicStruct::Builder data) override {
		auto builders = getBuilders(data.getSchema());
		
		size_t source = input.read(builders.size() + 1);
		if(source < builders.size()) {
			builders[source] -> fillStruct(data, *this);
			return;
		}
		
		auto setField = [&data, this](capnp::StructSchema::Field field) {
			if(field.getProto().isGroup()) {
				fillStruct(data.init(field).as<DynamicStruct>());
				return;
			}
			
			// Leave fields blank with 1/5 chance
			if(input.readOne() % 5 == 0)
				return;
			
			auto type = field.getType();
			
			if(type.isList()) {				
				uint32_t listSize = input.read(config.maxListSize);
				fillList(data.init(field, listSize).as<DynamicList>());
			} else if(type.isStruct()) {
				fillStruct(data.init(field).as<DynamicStruct>());
			} else if(type.isInterface()) {
				data.set(field, getCapability(type.asInterface()));
			} else if(type.isText()) {
				uint32_t blobSize = input.read(config.maxBlobSize);
				input.fill(data.init(field, blobSize).as<capnp::Text>().asBytes());
			} else if(type.isData()) {
				uint32_t blobSize = input.read(config.maxBlobSize);
				input.fill(data.init(field, blobSize).as<capnp::Data>());
			} else if(type.isAnyPointer()) {
				// Leave AnyPointer blank
			} else {
				// Read primitives directly from the wire
				data.set(field, input.readNonPointer(type));
			}
		};
		
		for(auto field : data.getSchema().getNonUnionFields()) {
			setField(field);
		}
		
		auto unionFields = data.getSchema().getUnionFields();
		if(unionFields.size() > 0) {
			uint16_t idx = input.read<uint16_t>(unionFields.size());
			setField(unionFields[idx]);
		}
	}
	
	void fillList(DynamicList::Builder data) {
		auto schema = data.getSchema();
		auto elType = schema.getElementType();
		
		if(elType.isList()) {
			for(auto i : kj::indices(data)) {
				uint32_t listSize = input.read(config.maxListSize);
				fillList(data.init(i, listSize).as<DynamicList>());
			}
		} else if(elType.isStruct()) {
			for(auto i : kj::indices(data)) {
				fillStruct(data[i].as<DynamicStruct>());
			}
		} else if(elType.isInterface()) {
			for(auto i : kj::indices(data)) {
				data.set(i, getCapability(elType.asInterface()));
			}
		} else if(elType.isText()) {
			for(auto i : kj::indices(data)) {
				uint32_t blobSize = input.read(config.maxBlobSize);
				input.fill(data.init(i, blobSize).as<capnp::Text>().asBytes());
			}
		} else if(elType.isData()) {
			for(auto i : kj::indices(data)) {
				uint32_t blobSize = input.read(config.maxBlobSize);
				input.fill(data.init(i, blobSize).as<capnp::Data>());
			}
		} else if(elType.isAnyPointer()) {
			// Leave AnyPointer blank
		} else {
			for(auto i : kj::indices(data)) {
				data.set(i, input.readNonPointer(elType));
			}
		}
	}
	
	void processPipeline(DynamicStruct::Pipeline& pipe) {
		auto schema = pipe.getSchema();
		
		// Collect candidate fields
		kj::Vector<StructSchema::Field> fields;
		for(auto field : schema.getNonUnionFields()) {
			if(field.getType().isStruct() || field.getType().isInterface()) {
				fields.add(field);
			}
		}
		
		if(fields.size() == 0)
			return;
		
		auto fieldIdx = input.read(fields.size());
		auto field = fields[fieldIdx];
				
		DynamicValue::Pipeline newPipe = pipe.get(fields[fieldIdx]);
		
		if(field.getType().isStruct()) {
			activePipeline = newPipe.releaseAs<DynamicStruct>();
		} else {
			addImport(newPipe.releaseAs<DynamicCapability>());
		}
	}		
	
	Promise<void> doCall() {
		// Pick a capability from the list
		if(imports.size() == 0)
			return READY_NOW;
		
		auto& import = *(input.selectFromList(imports));
		
		// Pick a method
		if(import.methods.size() == 0)
			return READY_NOW;
		
		auto& method = import.methods[input.read(import.methods.size())];
		
		// Make a call
		auto req = import.cap.newRequest(method);
		fillStruct(req);
		
		OutboundCall oc { req.send() };		
		outboundCalls.push_back(mv(oc));
		
		return READY_NOW;
	}
	
	Promise<void> doDelete() {
		if(imports.size() == 0)
			return READY_NOW;
		
		auto it = input.selectFromList(imports);
		imports.erase(it);
		
		return READY_NOW;
	}
	
	Promise<void> doCancel() {
		if(outboundCalls.size() == 0)
			return READY_NOW;
		
		auto it = input.selectFromList(outboundCalls);
		
		outboundCalls.erase(it);
		activePipeline = nullptr;
		
		return READY_NOW;
	}
	
	Promise<void> doAwait() {
		if(outboundCalls.size() == 0)
			return READY_NOW;
		
		auto it = input.selectFromList(outboundCalls);
		auto& call = *it;
		
		Promise<void> result = call.promise.then([this](capnp::Response<DynamicStruct> response) {
			handleIncoming(response);
		})
		.catch_([](kj::Exception e) {
			
		});
		
		outboundCalls.erase(it);
		activePipeline = nullptr;
		
		return result;
	}
	
	Promise<void> doPipe() {
		if(outboundCalls.size() == 0)
			return READY_NOW;
		
		auto& call = *input.selectFromList(outboundCalls);
		processPipeline(call.promise);
		
		return READY_NOW;
	}
	
	Promise<void> doExtendPipe() {
		KJ_IF_MAYBE(pPipe, activePipeline) {			
			processPipeline(*pPipe);
		}
		
		return READY_NOW;
	}
	
	Promise<void> doAccept() {
		if(exports.size() == 0)
			return READY_NOW;
		
		auto& exp = *input.selectFromList(exports);
		Promise<InboundCall> ibcPromise = exp.pImpl -> awaitCall(input.read<uint64_t>());
		
		return ibcPromise.then([this](InboundCall ibc) {
			handleIncoming(ibc.ctx.getParams());
			inboundCalls.push_back(mv(ibc));
		});
	}
	
	Promise<void> doFulfill() {
		if(inboundCalls.size() == 0)
			return READY_NOW;
		
		auto it = input.selectFromList(inboundCalls);
		auto& call = *it;
		
		fillStruct(call.ctx.initResults());
		call.fulfiller -> fulfill();
		
		inboundCalls.erase(it);
		
		return READY_NOW;
	}
	
	Promise<void> doReject() {
		if(inboundCalls.size() == 0)
			return READY_NOW;
		
		auto it = input.selectFromList(inboundCalls);
		auto& call = *it;
		
		Maybe<kj::Exception> exc;
		switch(input.read<uint8_t>(4)) {
			case 0: exc = KJ_EXCEPTION(FAILED, "Failed"); break;
			case 1: exc = KJ_EXCEPTION(OVERLOADED, "Overloaded"); break;
			case 2: exc = KJ_EXCEPTION(DISCONNECTED, "Disconnected"); break;
			case 3: exc = KJ_EXCEPTION(UNIMPLEMENTED, "Unimplemented"); break;
		}
		
		KJ_IF_MAYBE(pExc, exc) {
			call.fulfiller -> reject(mv(*pExc));
			inboundCalls.erase(it);
		}
		
		return READY_NOW;
	}
};

}

kj::Promise<void> runFuzzer(kj::ArrayPtr<const kj::byte> data, kj::ArrayPtr<capnp::DynamicCapability::Client> targets, ProtocolConfig config) {
	auto protoState = kj::heap<ProtocolState>(data, config);
	
	for(auto t : targets)
		protoState -> addImport(t);
	
	auto result = protoState -> run();
	return result.attach(mv(protoState));
}

}