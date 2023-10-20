#include "sqlite.h"

#include <sqlite3.h>
#include <kj/refcount.h>

namespace fsc { namespace db {

namespace {

struct SQLiteConnection : public Connection, kj::Refcounted {
	// Interface methods
	Own<Connection> addRef() override;
	Own<Connection> fork(bool readOnly) override;
	
	Own<PreparedStatementHook> prepareHook(kj::StringPtr sql) override;
	
	bool inTransaction() override;
	
	// Implementation
	SQLiteConnection(kj::StringPtr filename, bool readOnly);
	~SQLiteConnection();
	
	int check(int result);
	
	// Members
	kj::String filename;
	sqlite3* handle;
};

struct SQLiteStatementHook : public PreparedStatementHook {
	// Interface methods
	void reset() override;
	bool step() override;
	
	int64_t lastInsertedRowid() override;
	size_t nRowsModified() override;
	
	void setParameter(size_t, double) override;
	void setParameter(size_t, int64_t) override;
	void setParameter(size_t, kj::ArrayPtr<const byte>) override;
	void setParameter(size_t, kj::StringPtr) override;
	void setParameter(size_t, decltype(nullptr)) override;
	
	double getDouble(size_t) override;
	int64_t getInt64(size_t) override;
	kj::ArrayPtr<const byte> getBlob(size_t) override;
	kj::StringPtr getText(size_t) override;
	
	bool isNull(size_t) override;
	
	kj::String getColumnName(size_t) override;
	
	size_t size() override;
	
	// Implementation
	SQLiteStatementHook(SQLiteConnection& parent, kj::StringPtr sql);
	~SQLiteStatementHook();
	
	int check(int result) { return parent -> check(result); }
	
	// Members
	Own<SQLiteConnection> parent;
	sqlite3_stmt* handle;
	bool available = false;
};

// class SQLiteConnection

Own<Connection> SQLiteConnection::addRef() {
	return kj::addRef(*this);
}

Own<Connection> SQLiteConnection::fork(bool readOnly) {
	KJ_REQUIRE(!filename.startsWith(":memory:"), "Anonymous connections can not be forked");
	KJ_REQUIRE(!filename.startsWith("?"), "Anonymous connections can not be forked");
	KJ_REQUIRE(filename != "", "Anonymous connections can not be forked");
	
	return kj::refcounted<SQLiteConnection>(filename, readOnly);
}

Own<PreparedStatementHook> SQLiteConnection::prepareHook(kj::StringPtr sql) {
	return kj::heap<SQLiteStatementHook>(*this, sql);
}

SQLiteConnection::SQLiteConnection(kj::StringPtr filename, bool readOnly) :
	filename(kj::heapString(filename)),
	handle(nullptr)
{
	try {
		check(sqlite3_open_v2(
			filename.cStr(),
			&handle,
			(readOnly ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)) | SQLITE_OPEN_NOMUTEX | SQLITE_OPEN_EXRESCODE,
			nullptr
		));
	} catch(...) {
		sqlite3_close_v2(handle);
		throw;
	}
}

SQLiteConnection::~SQLiteConnection() {
	sqlite3_close_v2(handle);
}

int SQLiteConnection::check(int result) {
	if(result == SQLITE_OK)
		return result;
	
	if(result == SQLITE_ROW)
		return result;
	
	if(result == SQLITE_DONE)
		return result;
	
	if(result == SQLITE_BUSY) {
		kj::throwFatalException(KJ_EXCEPTION(OVERLOADED, "Database busy"));
	}
	
	int errorCode = sqlite3_errcode(handle);
	int extendedErrorCode = sqlite3_extended_errcode(handle);
	kj::String errorMessage = kj::str(sqlite3_errmsg(handle));
	
	KJ_FAIL_REQUIRE("SQL error in sqlite", errorCode, extendedErrorCode, errorMessage);
}

bool SQLiteConnection::inTransaction() {
	return sqlite3_get_autocommit(handle) == 0;
}

// class SQLiteStatementHook

void SQLiteStatementHook::reset() {
	available = false;
	check(sqlite3_reset(handle));
}

bool SQLiteStatementHook::step() {
	int result = check(sqlite3_step(handle));
	
	available = true;
	if(result == SQLITE_ROW)
		return true;
	
	available = false;
	if(result == SQLITE_DONE)
		return false;
	
	KJ_UNREACHABLE;
}

int64_t SQLiteStatementHook::lastInsertedRowid() {
	return sqlite3_last_insert_rowid(parent -> handle);
}

size_t SQLiteStatementHook::nRowsModified() {
	return sqlite3_changes64(parent -> handle);
}

void SQLiteStatementHook::setParameter(size_t idx, double d) {
	check(sqlite3_bind_double(handle, idx + 1, d));
}

void SQLiteStatementHook::setParameter(size_t idx, int64_t i) {
	check(sqlite3_bind_int64(handle, idx + 1, i));
}

void SQLiteStatementHook::setParameter(size_t idx, decltype(nullptr)) {
	check(sqlite3_bind_null(handle, idx + 1));
}

void SQLiteStatementHook::setParameter(size_t idx, ArrayPtr<const byte> blob) {
	check(sqlite3_bind_blob(handle, idx + 1, blob.begin(), blob.size(), SQLITE_TRANSIENT));
}

void SQLiteStatementHook::setParameter(size_t idx, kj::StringPtr text) {
	check(sqlite3_bind_text(handle, idx + 1, text.begin(), text.size(), SQLITE_TRANSIENT));
}

double SQLiteStatementHook::getDouble(size_t idx) {
	KJ_REQUIRE(available, "Statement has no active row");
	
	return sqlite3_column_double(handle, idx);
}

int64_t SQLiteStatementHook::getInt64(size_t idx) {
	KJ_REQUIRE(available, "Statement has no active row");
	
	return sqlite3_column_int64(handle, idx);
}

ArrayPtr<const byte> SQLiteStatementHook::getBlob(size_t idx) {
	KJ_REQUIRE(available, "Statement has no active row");
	
	return kj::ArrayPtr<const byte>(
		(const byte*) sqlite3_column_blob(handle, idx),
		sqlite3_column_bytes(handle, idx)
	);
}

kj::StringPtr SQLiteStatementHook::getText(size_t idx) {
	KJ_REQUIRE(available, "Statement has no active row");
	
	return kj::StringPtr(
		(const char*) sqlite3_column_text(handle, idx),
		sqlite3_column_bytes(handle, idx)
	);
}

bool SQLiteStatementHook::isNull(size_t idx) {
	KJ_REQUIRE(available, "Statement has no active row");
	
	return sqlite3_column_type(handle, idx) == SQLITE_NULL;
}

kj::String SQLiteStatementHook::getColumnName(size_t idx) {
	KJ_REQUIRE(available, "Statement has no active row");
	
	return kj::heapString(sqlite3_column_name(handle, idx));
}

size_t SQLiteStatementHook::size() {
	KJ_REQUIRE(available, "Statement has no active row");
	
	return sqlite3_column_count(handle);
}

SQLiteStatementHook::SQLiteStatementHook(SQLiteConnection& parent, kj::StringPtr sql) :
	parent(kj::addRef(parent))
{
	check(sqlite3_prepare_v2(
		parent.handle,
		sql.begin(), sql.size(),
		&handle,
		nullptr
	));
}

SQLiteStatementHook::~SQLiteStatementHook() {
	sqlite3_finalize(handle);
}

}

}}

namespace fsc {
	
Own<db::Connection> connectSqlite(kj::StringPtr url, bool readOnly) {
	auto result = kj::refcounted<db::SQLiteConnection>(url, readOnly);
	
	// Set up connection parameters
	result -> exec("PRAGMA journal_mode=WAL");
	result -> exec("PRAGMA foreign_keys = ON");
	
	return result;
}

}