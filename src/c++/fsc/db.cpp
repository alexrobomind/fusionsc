#include "db.h"
#include "sqlite3.h"

namespace fsc {

Own<SQLite3Connection> openSQLite3(kj::StringPtr filename) {
	return kj::refcounted<SQLite3Connection>(filename);
}
	
// ================= struct SQLite3Connection =================

void SQLite3Connection::check(int result) {
	if(result == SQLITE_OK)
		return;
	
	if(result == SQLITE_ROW)
		return;
	
	if(result == SQLITE_DONE)
		return;
	
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
		check(sqlite3_open(filename, &handle));
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

// ============= struct SQLite3PreparedStatement ==============

SQLite3PreparedStatement::SQLite3PreparedStatement(SQLite3Connection& conn, kj::StringPtr statement) :
	parent(conn.addRef()), handle(nullptr)
{
	check(sqlite3_prepare_v2(conn.handle, statement.begin(), statement.size()));
}

SQLite3PreparedStatement::~SQLite3PreparedStatement() {
	sqlite3_finalize(handle);
}

bool SQLite3PreparedStatement::step() {
	KJ_REQUIRE(state == ACTIVE, "Statement must be active, reinit using reset()");
	
	int retCode = sqlite3_step(handle);
	check(retCode);
	
	if(retCode == SQLITE_ROW)
		return true;
	
	state = DONE;
	return false;
}

void SQLite3PreparedStatement::reset() {
	check(sqlite3_reset(handle));
	state = ACTIVE;
}

void SQLite3PreparedStatement::size() {
	return sqlite3_column_count(handle);
}

int64_t SQLite3PreparedStatement::execInsert() {
	KJ_REQUIRE(state == DONE, "Statement must be done");
	
	step();
	reset();
	
	return parent -> lastInsertRowid();
}

int64_t SQLite3PreparedStatement::exec() {
	KJ_REQUIRE(state == DONE, "Statement must be done");
	
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
		(const byte*) sqlite3_column_text(parent.handle, idx),
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

// --- Parameter accessors ---

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(kj::ArrayPtr<const byte> blob) {
	check(sqlite3_bind_blob(parent.handle, idx, blob.begin(), blob.size(), SQLITE_TRANSIENT));
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(kj::StringPtr text) {
	check(sqlite3_bind_text(parent.handle, idx, text.begin(), text.size(), SQLITE_TRANSIENT));
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(double d) {
	check(sqlite3_bind_double(parent.handle, idx, d));
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(int i) {
	check(sqlite3_bind_int(parent.handle, idx, i));
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(int64_t i) {
	check(sqlite3_bind_int64(parent.handle, idx, i));
}

SQLite3PreparedStatement::Param& SQLite3PreparedStatement::Param::operator=(decltype(nullptr)) {
	check(sqlite3_bind_null(parent.handle));
}

}