#pragma once

#include "db.h"

namespace fsc {
	Own<db::Connection> connectSqlite(kj::StringPtr url, bool readOnly = false);
}
