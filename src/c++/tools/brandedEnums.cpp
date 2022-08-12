#include <capnp/schema.capnp.h>
#include <capnp/persistent.capnp.h>
#include <capnp/schema.h>
#include <capnp/orphan.h>

#include <capnp/message.h>

#include <kj/debug.h>

int main() {
	using namespace capnp;
	
	// Persistent<...>::SaveParams and ::SaveResults are the only generic types in the standard library
	using ListType = List<schema::ElementSize>;
	using Branded = Persistent<AnyPointer, ListType>::SaveParams;
	
	MallocMessageBuilder msg;
	auto root = msg.initRoot<Branded>();
	
	KJ_DBG(root); // Works
	
	auto data = root.initSealFor(1);
	data.set(0, schema::ElementSize::TWO_BYTES);
	
	KJ_DBG(root); // KABOOM
	
	return 0;
}