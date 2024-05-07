#pragma once

#include "common.h"
#include "data.h"

#include <kj/async-io.h>
#include <kj/compat/http.h>

#include <capnp/schema-loader.h>

#include <fsc/warehouse.capnp.h>

namespace fsc {

//! Creates an HTTP service that can be used to take a closer look at objects.
Own<kj::HttpService> createDataViewer(OneOf<DataRef<>::Client, Warehouse::Folder::Client, Warehouse::File<>::Client> root, capnp::SchemaLoader& loader);

}