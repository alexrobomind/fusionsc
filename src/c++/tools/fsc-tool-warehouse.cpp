#include <fsc/local.h>
#include <fsc/networking.h>
#include <fsc/services.h>
#include <fsc/sqlite.h>
#include <fsc/odb.h>

#include <capnp/rpc-twoparty.h>

#include <kj/main.h>
#include <kj/async.h>
#include <kj/thread.h>
#include <kj/compat/url.h>

#include <iostream>
#include <functional>

#include <capnp/ez-rpc.h>

#include "fsc-tool.h"

using namespace fsc;

namespace {

struct SimpleMessageFallback : public SimpleHttpServer::Server {
	Promise<void> serve(ServeContext ctx) {
		unsigned int UPGRADE_REQUIRED = 426;
		
		auto results = ctx.initResults();
		results.setStatus(UPGRADE_REQUIRED);
		results.setStatusText("Upgrade required");
		results.setBody(
			"This is a FusionSC warehouse, which expects WebSocket requests."
			" Please connect to it using the FusionSC client."
			" See: https://jugit.fz-juelich.de/a.knieps/fsc for more information"
		);
		return READY_NOW;
	}
};

struct WarehouseTool {
	kj::ProcessContext& context;
	Maybe<uint64_t> port = nullptr;
	kj::String address = kj::heapString("0.0.0.0");
	
	kj::String dbFile;
	kj::String tablePrefix;
	
	kj::String rootPath;
	
	kj::String backupFile;
	
	bool writeAccess = false;
	bool truncate = false;
	
	WarehouseTool(kj::ProcessContext& context):
		context(context), tablePrefix(kj::heapString("warehouse")), rootPath(kj::heapString(""))
	{}
	
	bool setAddress(kj::StringPtr val) {
		address = kj::str(val);
		return true;
	}
	
	bool setPort(kj::StringPtr val) {
		port = val.parseAs<uint64_t>();
		return true;
	}
	
	bool setWriteAccess() {
		writeAccess = true;
		return true;
	}
	
	bool setTruncate() {
		truncate = true;
		return true;
	}
	
	bool setDb(kj::StringPtr file) {
		dbFile = kj::heapString(file);
		return true;
	}
	
	bool setBackup(kj::StringPtr file) {
		backupFile = kj::heapString(file);
		return true;
	}
	
	bool setTablePrefix(kj::StringPtr prefix) {
		tablePrefix = kj::heapString(prefix);
		return true;
	}
	
	bool setPath(kj::StringPtr pathStr) {
		rootPath = kj::heapString(pathStr);
		return true;
	}
		
	bool serve() {
		auto l = newLibrary();
		auto lt = l -> newThread();
		auto& ws = lt->waitScope();
				
		// Open database
		
		bool readOnly = !writeAccess;
		auto conn = connectSqlite(dbFile, readOnly);
		auto db = ::fsc::openWarehouse(*conn, readOnly, tablePrefix);
		auto root = db.getRootRequest().sendForPipeline().getRoot();
		root.whenResolved().wait(ws);
		
		// Get root path
		if(readOnly) {
			auto getRequest = root.getRequest();
			getRequest.setPath(rootPath);
			
			auto rootObject = getRequest.send().wait(ws);
			KJ_REQUIRE(rootObject.isFolder(), "Requested root object is not a folder");
			
			root = rootObject.getFolder();
		} else {
			auto mkdirRequest = root.mkdirRequest();
			mkdirRequest.setPath(rootPath);
			root = mkdirRequest.send().wait(ws).getFolder();
		}
		
		// Create network interface
		NetworkInterface::Client networkInterface = kj::heap<LocalNetworkInterface>();
		
		auto serveRequest = networkInterface.serveRequest();
		KJ_IF_MAYBE(pPort, port) {
			serveRequest.setPortHint(*pPort);
		}
		serveRequest.setHost(address);
		serveRequest.setServer(root);
		serveRequest.setFallback(kj::heap<SimpleMessageFallback>());
		
		auto openPort = serveRequest.sendForPipeline().getOpenPort();
		auto info = openPort.getInfoRequest().send().wait(ws);
		
		std::cout << "Serving protocol version " << FSC_PROTOCOL_VERSION << std::endl;
		std::cout << "Listening on port " << info.getPort() << std::endl;
		
		while(true) {
			try {
				lt -> timer().afterDelay(1 * kj::SECONDS).wait(ws);
			} catch(kj::Exception e) {
				KJ_DBG("Received exception", e);
				break;
			}
		}
		
		std::cout << "Draining ..." << std::endl;
		openPort.drainRequest().send().wait(ws);
		
		std::cout << "Shutting down" << std::endl;		
		return true;
	}
	
	bool vacuum() {
		std::cout << "Opening database file " << dbFile.cStr() << std::endl;
		
		auto conn = connectSqlite(dbFile);
		conn -> exec("PRAGMA busy_timeout=5000");
		
		std::cout << "Beginning vacuum operation. Please note that this will lock any pending database writes." << std::endl;
		
		while(true) {
			try {
				conn -> exec("VACUUM");
				break;
			} catch(kj::Exception& e) {
				if(e.getType() == kj::Exception::Type::OVERLOADED) {
					KJ_LOG(WARNING, "Database is currently busy. Retrying.");
				} else {
					throw;
				}
			}
		}
		
		std::cout << "Vacuum complete." << std::endl;
		return true;
	}
	
	bool checkpoint() {
		std::cout << "Opening database file " << dbFile.cStr() << std::endl;
		
		auto conn = connectSqlite(dbFile);
		conn -> exec("PRAGMA busy_timeout=5000");
		
		std::cout << "Beginning checkpoint operation. Please note that this will lock any pending database writes." << std::endl;
		
		while(true) {
			try {
				if(truncate) {
					conn -> exec("PRAGMA main.wal_checkpoint(TRUNCATE)");
				} else {
					conn -> exec("PRAGMA main.wal_checkpoint(FULL)");
				}
				break;
			} catch(kj::Exception& e) {
				if(e.getType() == kj::Exception::Type::OVERLOADED) {
					KJ_LOG(WARNING, "Database is currently busy. Retrying.");
				} else {
					throw;
				}
			}
		}
		
		std::cout << "Checkpoint complete." << std::endl;
		return true;
	}
	
	bool backup() {
		std::cout << "Opening database file " << dbFile.cStr() << std::endl;
		
		auto conn = connectSqlite(dbFile);
		
		std::cout
			<< "Beginning backup operation. This operation will hold a read lock (which might grow the WAL size)." << std::endl
			<< "Target: " << backupFile.cStr() << std::endl
		;
		
		while(true) {
			try {
				auto stmt = conn -> prepare("VACUUM INTO ?");
				stmt(backupFile.asPtr());
				break;
			} catch(kj::Exception& e) {
				if(e.getType() == kj::Exception::Type::OVERLOADED) {
					KJ_LOG(WARNING, "Database is currently busy. Retrying.");
				} else {
					throw;
				}
			}
		}
		
		std::cout << "Backup complete." << std::endl;
		return true;
	}
	
	auto vacuumCmd() {
		return kj::MainBuilder(context, "", "Runs an sqlite VACUUM command to rebuild the database in-place. This command should be periodically"
			" executed in order to mitigate database fragmentation.")
			.expectArg("<database file>", KJ_BIND_METHOD(*this, setDb))
			.callAfterParsing(KJ_BIND_METHOD(*this, vacuum))
			.build()
		;	
	}
	
	auto checkpointCmd() {
		return kj::MainBuilder(context, "", "Writes all content in the write-ahead-log to disk. Usually, this operation is performed automatically during"
			" database operation. However, executing a large number of writes while having open database handles (especially to deleted objects, which are"
			" not synchronized to newer snapshots) can grow the WAL file to unreasonably large size. To combat this, run a checkpoint command with the"
			" --truncate option to clear the write-ahead queue and reset the file. Note that this is only possible if no past checkpoints are being held"
			" open. Otherwise, this command will block until that is the case.")
			.expectArg("<database file>", KJ_BIND_METHOD(*this, setDb))
			.addOption({"truncate"}, KJ_BIND_METHOD(*this, setTruncate), "Truncate the WAL file (blocks writing until no transaction reads from past history)")
			.callAfterParsing(KJ_BIND_METHOD(*this, checkpoint))
			.build()
		;	
	}
	
	auto backupCmd() {
		return kj::MainBuilder(context, "", "")
			.expectArg("<database file>", KJ_BIND_METHOD(*this, setDb))
			.expectArg("<backup file>", KJ_BIND_METHOD(*this, setBackup))
			.callAfterParsing(KJ_BIND_METHOD(*this, backup))
			.build()
		;
	}
	
	auto serveCmd() {
		return kj::MainBuilder(context, "", "")
			.addOptionWithArg({'a', "address"}, KJ_BIND_METHOD(*this, setAddress), "<address>", "Address to listen on, defaults to 0.0.0.0")
			.addOptionWithArg({'p', "port"}, KJ_BIND_METHOD(*this, setPort), "<port>", "Port to listen on, defaults to system-assigned")
			
			.addOption({'w', "write-access"}, KJ_BIND_METHOD(*this, setWriteAccess), "Enables write access to the target database")
			.addOptionWithArg({"table-prefix"}, KJ_BIND_METHOD(*this, setTablePrefix), "<prefix>", "Prefix to use for table names (default 'warehouse')")
			.addOptionWithArg({"path"}, KJ_BIND_METHOD(*this, setPath), "<path>", "Path to share relative to database root")
			.expectArg("<database file>", KJ_BIND_METHOD(*this, setDb))
			
			.callAfterParsing(KJ_BIND_METHOD(*this, serve))
			.build()
		;
	}
	
	auto getMain() {
		auto infoString = kj::str(
			"FusionSC warehouse manager\n",
			"Protocol version ", FSC_PROTOCOL_VERSION, "\n"
		);
		
		return kj::MainBuilder(context, infoString, "Manages a warehouse database")
			.addSubCommand("serve", KJ_BIND_METHOD(*this, serveCmd), "Serves a warehouse from an sqlite database")
			
			.addSubCommand("backup", KJ_BIND_METHOD(*this, backupCmd), "Create a backup of the database at target location.")
			.addSubCommand("vacuum", KJ_BIND_METHOD(*this, vacuumCmd), "Rebuilds the database in-place.")
			.addSubCommand("checkpoint", KJ_BIND_METHOD(*this, checkpointCmd), "Checkpoints the WAL to the database (and optionally truncates the WAL file)")
			.build()
		;
	}
};

}

fsc_tool::MainGen fsc_tool::warehouse(kj::ProcessContext& ctx) {
	return KJ_BIND_METHOD(WarehouseTool(ctx), getMain);
}