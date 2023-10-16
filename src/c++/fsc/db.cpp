#include "db.h"
#include "sqlite3.h"

namespace fsc { namespace db {
	
// class PreparedStatement
	
PreparedStatement::PreparedStatement(Own<PreparedStatementHook>&& hook) :
	hook(mv(hook))
{}
	
PreparedStatement::~PreparedStatement() {
}
	
bool PreparedStatement::step() {
	if(hook -> step()) {
		return true;
	} else {
		hook -> reset();
		return false;
	}
}

// class Savepoint
	
Savepoint::Savepoint(Connection& parent) :
	parent(parent.addRef()), id(parent.savepointCounter++)
{
	parent.prepare(kj::str("SAVEPOINT sp_", id))();
}

Savepoint::~Savepoint() {
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
}

// class Transaction

Transaction::Transaction(Connection& parent) :
		parent(parent.addRef())
{
	parent.beginTransaction();
}
	
Transaction::~Transaction() {
	if(!savepoint.active())
		return;
	
	if(ud.isUnwinding()) {
		try {
			rollback();
		} catch(...) {
		}
	} else {
		commit();
	}
}

}}