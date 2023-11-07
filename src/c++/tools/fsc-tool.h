#include <kj/main.h>

namespace fsc_tool {

using MainGen = kj::Function<kj::MainFunc()>;

MainGen loadBalancer(kj::ProcessContext&);
MainGen server(kj::ProcessContext&);
MainGen worker(kj::ProcessContext&);
MainGen warehouse(kj::ProcessContext&);
MainGen capnp(kj::ProcessContext&);

}