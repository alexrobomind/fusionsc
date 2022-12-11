#include "db.h"

namespace fsc {

struct BlobStore {
	template<kj::StringPtr stmt>
	using Statement = sqlite3::Statement;
	
	BlobStore(sqlite3::Connection& conn, kj::StringPtr tablePrefix) :
		beginTransaction(conn, "BEGIN TRANSACTION"),
		commit(conn, "COMMIT"),
		
		createBlob(conn, str("INSERT INTO ", tablePrefix, "_blobs")),
		setBlobHash(conn, str("UPDATE ", tablePrefix, "_blobs WHERE id = ? SET hash = ?")),
		findBlob(conn, str("SELECT id FROM ", tablePrefix, "_blobs WHERE hash = ?")),
		
		incRefExternal(conn, str("UPDATE ", tablePrefix, "_blobs WHERE id = ? SET externalRefcount = externalRefcount + 1")),
		decRefExternal(conn, str("UPDATE ", tablePrefix, "_blobs WHERE id = ? SET externalRefcount = externalRefcount - 1")),
		incRefInternal(conn, str("UPDATE ", tablePrefix, "_blobs WHERE id = ? SET internalRefcount = internalRefcount + 1")),
		decRefInternal(conn, str("UPDATE ", tablePrefix, "_blobs WHERE id = ? SET internalRefcount = internalRefcount - 1")),
		
		deleteIfOrphan(conn, str("DELETE FROM ", tablePrefix, "_blobs WHERE id = ? AND externalRefcount = 0 AND internalRefcount = 0")),
		
		createChunk(conn, str("INSERT INTO ", tablePrefix, "_chunks (id, chunkNo) VALUES (?, ?)"))
	{
		interpretSchema(tablePrefix);
	}
	
private:
	inline kj::Array<kj::String> schema(kj::StringPtr tableName);
};

kj::Array<kj::String> BlobStore::schema(kj::StringPtr tablePrefix) {
	return {
		str(
			"CREATE TABLE ", tablePrefix, "_blobs IF NOT EXISTS ("
			"  id INTEGER PRIMARY KEY,"
			"  hash BLOB UNIQUE," // SQLite UNIQUE allows multiple NULL values
			"  externalRefcount INTEGER,"
			"  internalRefcount INTEGER"
			")"
		),
		str(
			"CREATE TABLE ", tablePrefix, "_chunks IF NOT EXISTS ("
			"  id INTEGER,"
			"  chunkNo INTEGER,"
			"  data BLOB,"
			""
			"  FOREIGN KEY(id) REFERENCES ", tablePrefix, "_blobs(id) ON UPDATE CASCADE ON DELETE CASCADE"
			")"
		),
		
		str("CREATE INDEX IF NOT EXISTS ON ", tablePrefix, "_blobs (hash)"),
		str("CREATE INDEX IF NOT EXISTS ON ", tablePrefix, "_chunks (id, chunkNo)")
	};
}

}