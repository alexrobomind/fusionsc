#pragma once

#include "common.h"

#include <kj/refcount.h>

#include <utility>

// SQLite3 structs
struct sqlite3;
struct sqlite3_stmt;

namespace fsc {

struct SQLite3Connection;
struct SQLite3PreparedStatement;
struct SQLite3Blob;
struct SQLite3Savepoint;
struct SQLite3Transaction;
struct SQLite3RootTransaction;

enum class SQLite3Type;

namespace sqlite {
	using Connection = SQLite3Connection;
	using Statement = SQLite3PreparedStatement;
	using Blob = SQLite3Blob;
	using Savepoint = SQLite3Savepoint;
	using Transaction = SQLite3Transaction;
	using RootTransaction = SQLite3RootTransaction;
	using Type = SQLite3Type;
}

enum class SQLite3Type {
	INTEGER = 1,
	FLOAT = 2,
	TEXT = 3,
	BLOB = 4,
	NULLTYPE = 5
};

Own<SQLite3Connection> openSQLite3(kj::StringPtr filename);
	
struct SQLite3Connection : private kj::Refcounted {
	sqlite3* handle;
	
	~SQLite3Connection();
	
	void check(int retCode);
	Own<SQLite3Connection> addRef();
	
	SQLite3PreparedStatement prepare(kj::StringPtr statement);
	
	int64_t exec(kj::StringPtr statement);
	int64_t execInsert(kj::StringPtr statement);
	
	int64_t lastInsertRowid();
	int64_t nRowsModified();
	
	bool inTransaction();
	
	SQLite3Transaction beginTransaction(kj::StringPtr name = nullptr);
	SQLite3RootTransaction beginRootTransaction(bool immediate);
	Maybe<SQLite3RootTransaction> ensureTransaction(bool immediate);
	
private:
	uint64_t transactionUID = 0; //< Unique ID counter for transactions
	SQLite3Connection(kj::StringPtr filename);
	
	friend kj::Refcounted;
	
	template<typename T, typename... Params>
	friend Own<T> kj::refcounted(Params&&... params);
	
	template <typename T>
	friend Own<T> kj::addRef(T& object);
};

struct SQLite3PreparedStatement {
	/** Column accessor helper.
	 * 
	 * Meant to be use like "int iVal = statement[0];" or "auto iVal = statement[0].asInt();".
	 */
	struct Column {
		SQLite3PreparedStatement& parent;
		int idx;
		
		operator kj::ArrayPtr<const byte>();
		operator kj::StringPtr();
		operator double();
		operator int();
		operator int64_t();
		
		SQLite3Type type();
		
		inline kj::ArrayPtr<const byte> asBlob() { return operator kj::ArrayPtr<const byte>(); }
		inline double asDouble() { return operator double(); }
		inline int asInt() { return operator int(); }
		inline int64_t asInt64() { return operator int64_t(); }
		inline kj::StringPtr asText() { return operator kj::StringPtr(); }
		
		kj::String name();
		
		inline bool operator==(const Column& other) { return other.idx == this->idx; }
		inline bool operator!=(const Column& other) { return other.idx != this->idx; }
		inline Column& operator++() { ++idx; return *this; }
		
		inline Column* operator ->() { return this; }
		inline Column& operator  *() { return *this; }
	};
	
	/** Parameter accessor helper.
	 *
	 * Meant to be used like "int i = 0; statement.param(0) = i;"
	 */
	struct Param {
		SQLite3PreparedStatement& parent;
		int idx;
		
		Param& operator=(kj::ArrayPtr<const byte> blob);
		Param& operator=(kj::StringPtr text);
		Param& operator=(int intVal);
		Param& operator=(int64_t int64);
		Param& operator=(double doubleVal);
		Param& operator=(decltype(nullptr) nPtr);
		
		void setZero(size_t size);
	};
	
	SQLite3PreparedStatement(SQLite3Connection& conn, kj::StringPtr statement);
	
	inline SQLite3PreparedStatement() : handle(nullptr) {}
	~SQLite3PreparedStatement();
	
	SQLite3PreparedStatement(SQLite3PreparedStatement&&) = default;
	SQLite3PreparedStatement& operator=(SQLite3PreparedStatement&&) = default;
	
	inline Param param(int i) { return Param {*this, i}; }
	inline Column column(int i) { KJ_REQUIRE(state == ACTIVE); return Column {*this, i}; }
	
	inline Param operator[](int i) { return param(i); }
	
	struct Query {
		SQLite3PreparedStatement& parent;
		bool step();
		
		Query(SQLite3PreparedStatement& parent);
		Query(Query&& other);
		~Query();
	
		inline Column operator[](int i) { return parent.column(i); }
		inline Column begin() { return parent.column(0); }
		inline Column end() { return parent.column(parent.size()); }
		
	private:
		kj::UnwindDetector ud;
		bool movedFrom = false;
	};
	
	void reset();
	
	int size();
	
	int64_t exec();
	int64_t execInsert();
	
	template<typename... Params>
	void bind(Params... params) {
		bindInternal(std::index_sequence_for<Params...>(), params...);
	}
	
	template<typename... Params>
	int64_t operator()(Params... params) {
		bind(params...);
		return exec();
	}
	
	template<typename... Params>
	int64_t insert(Params... params) {
		bind(params...);
		return execInsert();
	}
	
	template<typename... Params>
	Query query(Params... params) {
		bind(params...);
		return Query(*this);
	}
	
	sqlite3_stmt * handle;
	Own<SQLite3Connection> parent;
	
private:
	enum {
		ACTIVE,
		DONE,
		READY
	} state = READY;
	
	void check(int retCode);
	
	template<typename... Params, size_t... indices>
	void bindInternal(std::integer_sequence<size_t, indices...> pIndices, Params... params) {
		int unused[] = {
			0, (this->param(indices + 1) = params, 1)...
		};
	}
	
	bool step();
};

/** Creates and maintains a savepoint. The savepoint is released without rollback on destruction.
 */
struct SQLite3Savepoint {
	SQLite3Savepoint(SQLite3Connection& conn, kj::StringPtr name);
	SQLite3Savepoint(const SQLite3Savepoint&) = delete;
	SQLite3Savepoint(SQLite3Savepoint&&) = default;
	
	~SQLite3Savepoint();
	
	void rollback();
	void release();
	
	bool isReleased() { return released; }
	
private:
	Own<SQLite3Connection> conn;
	kj::String name;
	bool released = false;
	
	kj::UnwindDetector ud;
};

/** Transaction class.
 *
 * Creates and maintains a savepoint that records all sql statements. Upon destruction, the savepoint
 * will be released if the object is destroyed normally, but will be rolled back upon an exception. 
 */
struct SQLite3Transaction {
	inline SQLite3Transaction(SQLite3Connection& conn, kj::StringPtr name) : savepoint(conn, name) {}
	SQLite3Transaction(SQLite3Transaction&&) = default;
	~SQLite3Transaction();
	
	inline void commit() { savepoint.release(); }
	inline void rollback() { savepoint.rollback(); }
	
private:
	SQLite3Savepoint savepoint;
	kj::UnwindDetector ud;
};

/** Transaction class.
 *
 * Creates and maintains a savepoint that records all sql statements. Upon destruction, the savepoint
 * will be released if the object is destroyed normally, but will be rolled back upon an exception. 
 */
struct SQLite3RootTransaction {
	inline SQLite3RootTransaction(SQLite3Connection& conn, bool immediate);
	SQLite3RootTransaction(SQLite3RootTransaction&&) = default;
	~SQLite3RootTransaction();
	
	void commit();
	void rollback();
	
private:
	Own<SQLite3Connection> conn;
	kj::UnwindDetector ud;
	bool active = true;
};

}