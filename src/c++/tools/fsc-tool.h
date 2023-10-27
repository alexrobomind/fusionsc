#include <kj/main.h>

namespace fsc_tool {

kj::MainFunc loadBalancer(kj::ProcessContext&);
kj::MainFunc server(kj::ProcessContext&);
kj::MainFunc worker(kj::ProcessContext&);
kj::MainFunc warehouse(kj::ProcessContext&);

}