#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <capnp/capability.h>

#include <fsc/local.h>

namespace fsc { namespace pybindings {	
	/**
	 * Starts a new worker thread, and creates a python object that
	 * can be used to connect to the capability return by this service.
	 *
	 * The object will be of type fusionsc.local.LocalServer, and each
	 * connect() call will forward to the connect() call of the LocalServer
	 * instance passed in.
	 *
	 * The fusionsc instance will not correspond to the instance managed by
	 * the main python thread as part of the fusionsc.native library, but will
	 * share the same local data store. All communication will go through a stable
	 * ABI, therefore the versions between the main fusionsc plugin and the version
	 * linked against this library can vary independently
	 */
	pybind11::object createLocalServer(kj::Function<capnp::Capability::Client()> service, capnp::InterfaceSchema = capnp::Schema::from<capnp::Capability>());
	
	/**
	 * Creates a FusionSC library instance sharing its data store with the main
	 * FusionSC library registered in the python plugin.
	 */
	Library newPythonBoundLibrary();
}}