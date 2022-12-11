#include <utility>

// SQLite3 structs
struct sqlite3;
struct sqlite3_stmt;

namespace fsc {

struct SQLite3Connection;
struct SQLite3PreparedStatement;
struct SQLite3Blob;

namespace sqlite3 {
	using Connection = SQLite3Connection;
	using Statement = SQLite3PreparedStatement;
	using Blob = SQLite3Blob;
}

Own<SQLite3Connection> openSQLite3(kj::StringPtr filename);
	
struct SQLite3Connection : private kj::Refcounted {
	sqlite3* handle;
	
	~SQLite3Connection();
	
	void check();
	Own<SQLite3Connection> addRef();
	
	SQLite3PreparedStatement prepare(kj::StringPtr statement);
	
	int64_t lastInsertRowid();
	int64_t nRowsModified();
	
private:
	SQLite3Connection(kj::StringPtr filename);
	friend Own<SQLite3Connection> openSQLite3(kj::StringPtr filename);
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
		inline Column& operator  *() { return this; }
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
		Param& operator=(int64_t int64);
		Param& operator=(double doubleVal);
		Param& operator=(decltype(nullptr) nPtr);
		
		void setZero(size_t size);
	};
	
	SQLite3PreparedStatement(SQLite3Connection& conn, kj::StringPtr statement);
	~SQLite3PreparedStatement();
	
	SQLite3PreparedStatement(SQLite3PreparedStatement&&) = default;
	SQLite3PreparedStatement& operator=(SQLite3PreparedStatement&&) = default;
	
	bool step();
	void reset();
	
	int size();
	
	int64_t exec();
	int64_t execInsert();
	
	template<typename... Params>
	void bind(Params... params) {
		bindInternal(params..., std::index_sequence_for(params...));
	}
	
	template<typename... Params>
	int64_t operator()(Params... params) {
		bind(params...);
		return exec();
	}
	
	template<typename... Params>
	int64_t insert()(Params... params) {
		bind(params...);
		return execInsert();
	}
	
	inline Column operator[](int i) { return column(i); }
	inline Column begin() { return column(0); }
	inline Column end() { return column(size()); }
	
	inline Param param(int i) { return Param {*this, i}; }
	inline Column column(int i) { return Column {*this, i}; }
	
	sqlite3_stmt * handle;
	Own<SQLite3Connection> parent;	
private:
	enum {
		ACTIVE,
		DONE
	} state = ACTIVE;
	
	void check(int retCode);
	
	template<typename... Params, size_t... indices>
	bindAll(Params... params, std::integer_sequence<indices...>) {
		int unused[] = {
			(this->operator[](indices) = params, 1)...
		};
	}
};

}