#pragma once

#include "common.h"

namespace fsc { namespace db {

struct PreparedStatementHook {
	virtual void reset() = 0;
	virtual bool step() = 0;
	
	virtual int64_t lastInsertedRowid() = 0;
	virtual size_t nRowsModified() = 0;
	
	virtual void setParameter(size_t, double) = 0;
	virtual void setParameter(size_t, int64_t) = 0;
	virtual void setParameter(size_t, kj::ArrayPtr<const byte>) = 0;
	virtual void setParameter(size_t, kj::StringPtr) = 0;
	virtual void setParameter(size_t, decltype(nullptr)) = 0;
	
	virtual double getDouble(size_t) = 0;
	virtual int64_t getInt64(size_t) = 0;
	virtual kj::ArrayPtr<const byte> getBlob(size_t) = 0;
	virtual kj::StringPtr getText(size_t) = 0;
	
	virtual kj::String getColumnName(size_t) = 0;
	virtual size_t size() = 0;
};

struct PreparedStatement {
	PreparedStatement(Own<PreparedStatementHook>&& hook);
	~PreparedStatement();
	
	template<typename P>
	void setParameter(size_t, P);
	
	//! Assigns all parameters from arguments
	template<typename... Params>
	void bind(Params... params);
	
	struct Column;
	Column operator[](size_t idx);
	
	bool step();
	
	inline bool reset() { hook -> reset(); }
	
	template<typename... Params>
	size_t operator()(Params... params) {
		bind(params...);
		
		hook -> step();
		hook -> reset();
		
		return hook -> nRowsModified();
	}
	
	template<typename... Params>
	int64_t insert(Params... params) {
		bind(params...);
		
		hook -> step();
		hook -> reset();
		
		return hook -> lastInsertedRowid();
	}
	
private:
	//! Helper method for binding with known index sequence
	template<typename... Params, size_t... indices>
	void bindInternal(std::integer_sequence<size_t, indices...> pIndices, Params... params);
	
	Own<PreparedStatementHook> hook;
};

struct PreparedStatement::Column {
	PreparedStatement& parent;
	size_t idx;
	
	inline Column(PreparedStatement& parent, size_t idx) :
		parent(parent), idx(idx)
	{}
	
	inline kj::ArrayPtr<const byte> asBlob() { return parent.hook -> getBlob(idx); }
	inline double asDouble() { return parent.hook -> getDouble(idx); }
	inline int64_t asInt64() { return parent.hook -> getInt64(idx); }
	inline kj::StringPtr asText() { return parent.hook -> getText(idx); }
	
	inline operator kj::ArrayPtr<const byte>() { return asBlob(); }
	inline operator kj::StringPtr() { return asText(); }
	inline operator double() { return asDouble(); }
	inline operator int64_t() { return asInt64() ; }
	
	inline kj::String name() { return parent.hook -> getColumnName(idx); }
	
	inline bool operator==(const Column& other) { return other.idx == this->idx; }
	inline bool operator!=(const Column& other) { return other.idx != this->idx; }
	inline Column& operator++() { ++idx; return *this; }
	
	inline Column* operator ->() { return this; }
	inline Column& operator  *() { return *this; }
};

struct Connection {
	virtual Own<Connection> addRef() = 0;
	virtual Own<Connection> fork(bool readOnly) = 0;
	
	virtual Own<PreparedStatementHook> prepareHook(kj::StringPtr sql) = 0;
	
	inline PreparedStatement prepare(kj::StringPtr sql) { return prepareHook(sql); }
	inline int64_t exec(kj::StringPtr sql) { return prepare(sql)(); }

private:
	uint64_t savepointCounter;
	friend class Savepoint;
};

//! A savepoint that can be acquired and later released (default behavior) or rolled back
struct Savepoint {
	Own<Connection> parent;
	uint64_t id;
	kj::UnwindDetector ud;
	
	Savepoint(Connection& parent);
	~Savepoint();
	
	bool active();
	void release();
	void rollback();
};

//! A savepoint that will roll back on stack unwind
struct Transaction {
	Savepoint savepoint;
	
	Transaction(Connection& parent);
	~Transaction();
	
	inline void commit() {savepoint.release(); }
	inline void rollback() { savepoint.rollback(); }
};

}}

// Implementation


namespace fsc { namespace db {

PreparedStatement::Column PreparedStatement::operator[](size_t idx) {
	return Column(*this, idx);
}

template<typename P>
void PreparedStatement::setParameter(size_t idx, P p) {
	hook -> setParameter(idx, p);
}

template<typename... Params>
void PreparedStatement::bind(Params... params) {
	bindInternal(std::index_sequence_for<Params...>(), params...);
}
	
template<typename... Params, size_t... indices>
void bindInternal(std::integer_sequence<size_t, indices...> pIndices, Params... params) {
	int unused[] = {
		0, (hook -> setParameter(indices, params), 1)...
	};
}
	
}}