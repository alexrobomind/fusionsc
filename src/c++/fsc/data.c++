namespace fsc {

namespace {

class PromiseGetter : DataRef<capnp::AnyPointer>::Getter::Server {
public:
	using Promise = kj::Promise<capnp::AnyPointer::Builder>;
	using ForkedPromise = kj::ForkedPromise<capnp::AnyPointer::Builder>;
	
	PromiseGetter(Promise&& p) : promise(p.fork()), called(False) {}
	
	kj::Promise<void> get(GetContext ctx) {
		KJ_REQUIRE()
		return promise.addBranch().then([](capnp::AnyPointer::Builder result) {
			auto output = ctx.initResults();
			output.setValue(result);
		});
	}
private:
	ForkedPromise promise;
};

class SecureDataPipeImpl : public SecureDataPipe::Server {
public:
	::kj::Promise<void> read (ReadContext  context) override;
	::kj::Promise<void> write(WriteContext context) override;
	
	SecureDataPipeImpl(DataIDGenerator::Client newGen) : idGenerator(newGen) {}
	
private:
	using Key = kj::Array<byte>;
	struct Row {
		Key id;
		Own<kj::PromiseFulfiller<capnp::AnyPointer::Builder>> target;
	};
	
	class Callbacks {
	public:
		inline Key keyForRow (const Row& r) { return r->key(); }
		inline bool isBefore (const Row& r, const Key& k) { return internal::compareDSKeys(r.id, k) < 0; }
		inline bool matches  (const Row& r, const Key& k) { return internal::compareDSKeys(r.id, k) == 0; }
	};
	
	DataIDGenerator::Client idGenerator;
	::kj::Table<Row, ::kj::TreeIndex<Callbacks>> table;
};

::kj::Promise<void> SecureDataPipeImpl::read(ReadContext ctx) {
	kj::Array<byte> id();
	
	//TODO: Generate
	do {
		
	} while(table.find(id) != nullptr)
	
	// Generate promise / fulfiller pair
	auto paf = kj::newPromiseAndFulfiller<capnp::AnyPointer::Builder>();
	
	// Store id and fulfiller in table
	Row row;
	row.id = kj::HeapArray(id);
	row.target = kj::mv(paf.fulfiller);
	table.insert(kj::mv(row));
	
	// Pass ID and fulfiller back to client
	auto result = ctx.initResults();
	
	result.setId(id);
	result.setGetter(kj::heap<PromiseGetter>(kj::mv(paf.promise)));
}

::kj::Promise<void> SecureDataPipeImpl::write(WriteContext ctx) {
	auto id = ctx.getParams().getId().asPtr();
	
	// Check if we have corresponding row in table
	kj::Maybe<Row>& mrow = table.find(id);
	
	// If found, fulfill the target
	KJ_IF_MAYBE(mrow, row) {
		row.target -> fulfill(ctx.getParams().getData());
	}
}

}

kj::Own<SecureDataPipe::Server> secureDataPipeServer(DataIDGenerator::Client idGenerator) {
	return kj::heap<SecureDataPipeImpl>(idGenerator);
}

}