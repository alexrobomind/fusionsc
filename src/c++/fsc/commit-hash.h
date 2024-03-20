#pragma once

#include <kj/string.h>

namespace fsc {

// The following value is extracted - if possible - by the build system using
// a git command. The corresponding source file commit-hash is auto-generated.
extern kj::StringPtr commitHash;

}