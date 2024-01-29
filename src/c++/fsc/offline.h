#pragma once

#include <fsc/offline.capnp.h>
#include <fsc/magnetics.capnp.h>
#include <fsc/geometry.capnp.h>

namespace fsc {

FieldResolver::Client newOfflineFieldResolver(DataRef<OfflineData>::Client data);
GeometryResolver::Client newOfflineGeometryResolver(DataRef<OfflineData>::Client data);

void updateOfflineData(OfflineData::Builder, OfflineData::Reader);

}