#include "db.h"
#include "sqlite3.h"

namespace fsc { namespace db {

namespace {

// A transaction hook that is slaved to a Connection::BaseTransactionHook and manages
// a savepoint state.
struct SavepointTransactionHook : public Connection::TransactionHook {
	kj::ListLink<SavepointTransactionHook> listLink;
	Own<Connection::BaseTransactionHook> base;
	
	uint64_t id;
	
	SavepointTransactionHook(Connection::BaseTransactionHook& parent, uint64_t id);
	~SavepointTransactionHook() noexcept { deactivate(); };
	
	bool active() noexcept override { return listLink.isLinked(); }
	
	void commit() override;
	void rollback() noexcept override;
	
	void deactivate() noexcept;
};

}

struct Connection::BaseTransactionHook : public Connection::TransactionHook, kj::Refcounted {
	kj::List<SavepointTransactionHook, &SavepointTransactionHook::listLink> savepoints;
	Own<Connection> connection;
	
	uint64_t savepointCounter = 0;
	
	BaseTransactionHook(Connection& c) : connection(c.addRef()) {};
	~BaseTransactionHook() noexcept {
		KJ_ASSERT(!active(), "Transaction hook deleted before commit / rollback");
		
		// For safety reasons
		deactivate();
	};
	
	Own<Connection::BaseTransactionHook> addRef() { return kj::addRef(*this); }
	
	void deactivate() noexcept {
		if(!active()) return;
		
		for(SavepointTransactionHook& sp : savepoints) {
			sp.deactivate();
		}
		connection -> activeTransaction = nullptr;
	}
	
	bool active() noexcept override {
		KJ_IF_MAYBE(pConn, connection -> activeTransaction) {
			return pConn -> get() == this;
		} else {
			return false;
		}
	}
	
	void commit() override {
		deactivate();
		connection -> exec("COMMIT");
	}
	
	void rollback() noexcept override {
		deactivate();
		
		try {
			connection -> exec("ROLLBACK");
		} catch(kj::Exception& e) {
			KJ_LOG(WARNING, "Error during transaction rollback", e);
		}
	}
	
	Own<SavepointTransactionHook> newSavepoint() {
		uint64_t id = savepointCounter++;
		connection -> exec(kj::str("SAVEPOINT \"", id, "\""));
		
		return kj::heap<SavepointTransactionHook>(*this, id);
	}
};

SavepointTransactionHook::SavepointTransactionHook(Connection::BaseTransactionHook& parent, uint64_t id) :
	base(parent.addRef()),
	id(id)
{
	parent.savepoints.add(*this);
}

void SavepointTransactionHook::deactivate() noexcept {
	if(listLink.isLinked()) { base -> savepoints.remove(*this); }
}
	

void SavepointTransactionHook::commit() {
	base -> connection -> exec(kj::str("RELEASE SAVEPOINT \"", id, "\""));
	deactivate();
}

void SavepointTransactionHook::rollback() noexcept {
	try {
		base -> connection -> exec(kj::str("ROLLBACK TO \"", id, "\""));
	} catch(kj::Exception& e) {
		KJ_LOG(WARNING, "Failure during savepoint rollback", e);
	}
	
	deactivate();
}

Own<Connection::TransactionHook> Connection::beginTransactionBase(kj::StringPtr beginStatement) {
	KJ_IF_MAYBE(pHook, activeTransaction) {
		return (**pHook).newSavepoint();
	}
	
	exec(beginStatement);
	return activeTransaction.emplace(kj::refcounted<BaseTransactionHook>(*this)) -> addRef();
}	
	
// class TransactionHook

Connection::TransactionHook::~TransactionHook() {}
	
// class PreparedStatement
	
PreparedStatement::PreparedStatement(Own<PreparedStatementHook>&& hook) :
	hook(mv(hook))
{}
	
PreparedStatement::~PreparedStatement() {
}
	
bool PreparedStatement::Query::step() {
	if(parent.hook -> step()) {
		return true;
	} else {
		parent.hook -> reset();
		return false;
	}
}

// class Savepoint
	
/*Savepoint::Savepoint(Connection& parent) :
	parent(parent.addRef()), id(parent.savepointCounter++)
{
	parent.prepare(kj::str("SAVEPOINT sp_", id))();
}

Savepoint::~Savepoint() noexcept(false) {
	if(active())
		ud.catchExceptionsIfUnwinding([this]() { release(); });
}

bool Savepoint::active() {
	return parent.get() != nullptr;
}

void Savepoint::release() {
	KJ_REQUIRE(active(), "Savepoint must be active to release");
	parent -> prepare(kj::str("RELEASE sp_", id))();
	parent = nullptr;
}

void Savepoint::rollback() {
	KJ_REQUIRE(active(), "Savepoint must be active to roll back");
	parent -> prepare(kj::str("ROLLBACK TO sp_", id))();
	parent = nullptr;
}*/

// class Transaction
/*
Transaction::Transaction(Connection& parent) :
	savepoint(parent)
{}
	
Transaction::~Transaction() noexcept(false) {
	if(!savepoint.active())
		return;
	
	if(savepoint.ud.isUnwinding()) {
		try {
			rollback();
		} catch(...) {
		}
	} else {
		commit();
	}
}*/

Transaction::Transaction(Connection& parent, TransactionType type) :
	hook(parent.beginTransaction(type))
{}

Transaction::~Transaction() noexcept(false) {
	if(!hook -> active())
		return;
	
	if(ud.isUnwinding()) {
		hook -> rollback();
	} else {
		hook -> commit();
	}
}

}}