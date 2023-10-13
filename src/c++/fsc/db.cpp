#include "db.h"
#include "sqlite3.h"

namespace fsc {

Own<SQLite3Connection> openSQLite3(kj::StringPtr filename) {
	auto result = kj::refcounted<SQLite3Connection>(filename);
	result -> exec("PRAGMA foreign_keys = ON");
	return result;
}
	
// ================= struct SQLite3Connection =================

void SQLite3Connection::check(int result) {
	if(result == SQLITE_OK)
		return;
	
	if(result == SQLITE_ROW)
		return;
	
	if(result == SQLITE_DONE)
		return;
	
	if(result == SQLITE_BUSY) {
		kj::throwFatalException(KJ_EXCEPTION(OVERLOADED, "Database busy"));
	}
	
	int errorCode = sqlite3_errcode(handle);
	int extendedErrorCode = sqlite3_extended_errcode(handle);
	kj::String errorMessage = kj::str(sqlite3_errmsg(handle));
	
	KJ_FAIL_REQUIRE("SQL error in sqlite", errorCode, extendedErrorCode, errorMessage);
}

Own<SQLite3Connection> SQLite3Connection::addRef() {
	return kj::addRef(*this);
}

SQLite3Connection::SQLite3Connection(kj::StringPtr filename) :
	handle(nullptr)
{
	try {
		check(sqlite3_open(filename.cStr(), &handle));
	} catch(kj::Exception& e) {
		sqlite3_close_v2(handle);
		throw e;
	}
}

SQLite3Connection::~SQLite3Connection() {
	sqlite3_close_v2(handle);
}

int64_t SQLite3Connection::lastInsertRowid() {
	return sqlite3_last_insert_rowid(handle);
}

int64_t SQLite3Connection::nRowsModified() {
	return sqlite3_changes64(handle);
}

bool SQLite3Connection::inTransaction() {
	return sqlite3_get_autocommit(handle) == 0;
}
	
int64_t SQLite3Connection::exec(kj::StringPtr statement) {
	return prepare(statement).exec();
}

int64_t SQLite3Connection::execInsert(kj::StringPtr statement) {
	return prepare(statement).execInsert();
}

SQLite3Transaction SQLite3Connection::beginTransaction(kj::StringPtr name) {
	kj::String privateName;
	
	if(name.size() == 0) {
		privateName = kj::str("fsc_unique_transaction_", transactionUID++);
		name = privateName;
	}
	
	return SQLite3Transaction(*this, name);
}

SQLite3RootTransaction SQLite3Connection::beginRootTransaction (bool immediate) {
	return SQLite3RootTransaction(*this, immediate);
}

Maybe<SQLite3RootTransaction> SQLite3Connection::ensureTransaction(bool immediate) {
	if(inTransaction())
		return nullptr;
	return beginRootTransaction(immediate);
}

SQLite3PreparedStatement SQLite3Connection::prepare(kj::StringPtr statement) {
	// KJ_DBG("Preparing statement statement", statement);
	return SQLite3PreparedStatement(*this, statement);
}

// ============= struct SQLite3PreparedStatement ==============

SQLite3PreparedStatement::SQLite3PreparedStatement(SQLite3Connection& conn, kj::StringPtr statement) :
	parent(conn.addRef()), handle(nullptr)
{
	KJ_ASSERT(&conn != nullptr);
	check(sqlite3_prepare_v2(conn.handle, statement.begin(), statement.size(), &handle, nullptr));
}

SQLite3PreparedStatement::~SQLite3PreparedStatement() {
	if(parent.get() != nullptr) {
		sqlite3_finalize(handle);
	}
}

bool SQLite3PreparedStatement::step() {
	KJ_REQUIRE(state == ACTIVE || state == READY, "Statement must be ready or active");
	
	int retCode = sqlite3_step(handle);
	check(retCode);
	
	if(retCode == SQLITE_ROW) {
		state = ACTIVE;
		return true;
	}
	
	state = DONE;
	return false;
}

void SQLite3PreparedStatement::reset() {
	check(sqlite3_reset(handle));
	state = READY;
}

int SQLite3PreparedStatement::size() {
	return sqlite3_column_count(handle);
}

int64_t SQLite3PreparedStatement::execInsert() {
	KJ_REQUIRE(state == READY, "Statement must be ready");
	
	step();
	reset();
	
	return parent -> lastInsertRowid();
}

int64_t SQLite3PreparedStatement::exec() {
	KJ_REQUIRE(state == READY, "Statement must be ready");
	
	step();
	reset();
	
	return parent -> nRowsModified();
}

void SQLite3PreparedStatement::check(int retCode) {
	parent -> check(retCode);
}

// --- Column accessors ---

SQLite3PreparedStatement::Column::operator kj::ArrayPtr<const byte>() {
	KJ_REQUIRE(parent.state == ACTIVE, "Statement must be active");
	
	return kj::ArrayPtr<const byte>(
		(const byte*) sqlite3_column_blob(parent.handle, idx),
		sqlite3_column_bytes(parent.handle, idx)
	);
}

SQLite3PreparedStatement::Column::operator kj::StringPtr() {
	KJ_REQUIRE(parent.state == ACTIVE, "Statement must be active");
	
	return kj::StringPtr(
		(const char*) sqlite3_column_text(parent.handle, idx),
		sqlite3_column_bytes(parent.handle, idx)
	);
}

SQLite3PreparedStatement::Column::operator double() {
	KJ_REQUIRE(parent.state == ACTIVE, "Statement must be active");
	
	return sqlite3_column_double(parent.handle, idx);
}

SQLite3PreparedStatement::Column::operator int() {
	KJ_REQUIRE(parent.state == ACTIVE, "Statement must be active");
	
	return sqlite3_column_int(parent.handle, idx);
}

SQLite3PreparedStatement::Column::operator int64_t() {
	KJ_REQUIRE(parent.state == ACTIVE, "Statement must be active");
	
	return sqlite3_column_int64(parent.handle, idx);
}

kj::String SQLite3PreparedStatement::Column::name() {	
	return kj::heapString(sqlite3_column_name(parent.handle, idx));
}

SQLite3Type SQLite3PreparedStatement::Column::type() {	
	return (SQLite3Type) sqlite3_column_type(parent.handle, idx);
}

// --- Parameter accessors ---

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(kj::ArrayPtr<const byte> blob) {
	parent.check(sqlite3_bind_blob(parent.handle, idx, blob.begin(), blob.size(), SQLITE_TRANSIENT));
	return *this;
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(kj::StringPtr text) {
	parent.check(sqlite3_bind_text(parent.handle, idx, text.begin(), text.size(), SQLITE_TRANSIENT));
	return *this;
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(double d) {
	parent.check(sqlite3_bind_double(parent.handle, idx, d));
	return *this;
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(int i) {
	parent.check(sqlite3_bind_int(parent.handle, idx, i));
	return *this;
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(int64_t i) {
	parent.check(sqlite3_bind_int64(parent.handle, idx, i));
	return *this;
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(decltype(nullptr)) {
	parent.check(sqlite3_bind_null(parent.handle, idx));
	return *this;
}

SQLite3PreparedStatement::Query::Query(SQLite3PreparedStatement& parent) :
	parent(parent)
{
	KJ_REQUIRE(parent.state == READY);
	parent.state = ACTIVE;
}

SQLite3PreparedStatement::Query::Query(Query&& other) :
	parent(other.parent)
{
	other.movedFrom = true;
}

SQLite3PreparedStatement::Query::~Query() {
	if(movedFrom)
		return;
	
	ud.catchExceptionsIfUnwinding([this]() {
		parent.reset();
	});
}

bool SQLite3PreparedStatement::Query::step() {
	return parent.step();
}

// ============================== SQLite3Savepoint =============================

SQLite3Savepoint::SQLite3Savepoint(SQLite3Connection& conn, kj::StringPtr name) :
	conn(conn.addRef()),
	name(kj::heapString(name))
{
	// KJ_DBG("Creating savepoint", name);
	conn.exec(str("SAVEPOINT ", name));
}

SQLite3Savepoint::~SQLite3Savepoint() {
	ud.catchExceptionsIfUnwinding([this]{ release(); });
}

void SQLite3Savepoint::rollback() {
	KJ_REQUIRE(!released, "Trying to roll back released savepoint");
	// KJ_DBG("Rolling back savepoint", name);
	
	if(conn.get() == nullptr)
		return;
	
	conn -> exec(str("ROLLBACK TO ", name));
}

void SQLite3Savepoint::release() {		
	if(!released) {
		if(conn.get() != nullptr)
			conn -> exec(str("RELEASE SAVEPOINT ", name));
		
		released  = true;
		// KJ_DBG("Releasing savepoint", name);
	}
}

// =========================== SQLite3Transaction =======================

SQLite3Transaction::~SQLite3Transaction() {
	if(ud.isUnwinding()) {
		if(!savepoint.isReleased()) {
			ud.catchExceptionsIfUnwinding([this] { savepoint.rollback(); });
		}
	}
}

// =========================== SQLite3RootTransaction =======================

SQLite3RootTransaction::SQLite3RootTransaction(SQLite3Connection& conn, bool immediate) :
	conn(conn.addRef())
{
	KJ_REQUIRE(!conn.inTransaction(), "Root transactions can only be started outside any transaction");
	if(immediate)
		conn.exec("BEGIN IMMEDIATE TRANSACTION");
	else
		conn.exec("BEGIN TRANSACTION");
}

SQLite3RootTransaction::~SQLite3RootTransaction() {	
	if(conn.get() == nullptr)
		return;
	
	if(ud.isUnwinding()) {
		kj::runCatchingExceptions([this]() { rollback(); });
	} else {
		commit();
	}
}

void SQLite3RootTransaction::commit() {
	conn -> exec("COMMIT");
}

void SQLite3RootTransaction::rollback() {
	conn -> exec("ROLLBACK");
}

}