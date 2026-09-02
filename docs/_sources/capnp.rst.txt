Cap'n'Proto Schema Documentation
================================

This section documents the Cap'n'Proto schemas used in FusionSC.
These schemas define the data structures and RPC interfaces for the
distributed services.

.. toctree::
   :maxdepth: 2
   :caption: Schema Files:

   capnp/data-archive
   capnp/data-test
   capnp/data
   capnp/dynamic
   capnp/flt
   capnp/geometry-test
   capnp/geometry
   capnp/hfcam
   capnp/hint
   capnp/http
   capnp/index
   capnp/java
   capnp/jobs
   capnp/local-vat-network
   capnp/magnetics-test
   capnp/magnetics
   capnp/matcher
   capnp/networking
   capnp/offline
   capnp/random
   capnp/services
   capnp/streams
   capnp/vmec
   capnp/warehouse-internal
   capnp/warehouse

.. toctree::
   :maxdepth: 1
   :caption: API Examples:

   capnp/data-archive_cpp
   capnp/data-archive_python
   capnp/data-test_cpp
   capnp/data-test_python
   capnp/data_cpp
   capnp/data_python
   capnp/dynamic_cpp
   capnp/dynamic_python
   capnp/flt_cpp
   capnp/flt_python
   capnp/geometry-test_cpp
   capnp/geometry-test_python
   capnp/geometry_cpp
   capnp/geometry_python
   capnp/hfcam_cpp
   capnp/hfcam_python
   capnp/hint_cpp
   capnp/hint_python
   capnp/http_cpp
   capnp/http_python
   capnp/index_cpp
   capnp/index_python
   capnp/java_cpp
   capnp/java_python
   capnp/jobs_cpp
   capnp/jobs_python
   capnp/local-vat-network_cpp
   capnp/local-vat-network_python
   capnp/magnetics-test_cpp
   capnp/magnetics-test_python
   capnp/magnetics_cpp
   capnp/magnetics_python
   capnp/matcher_cpp
   capnp/matcher_python
   capnp/networking_cpp
   capnp/networking_python
   capnp/offline_cpp
   capnp/offline_python
   capnp/random_cpp
   capnp/random_python
   capnp/services_cpp
   capnp/services_python
   capnp/streams_cpp
   capnp/streams_python
   capnp/vmec_cpp
   capnp/vmec_python
   capnp/warehouse-internal_cpp
   capnp/warehouse-internal_python
   capnp/warehouse_cpp
   capnp/warehouse_python

Overview
--------

FusionSC uses Cap'n'Proto for:

- **Data serialization**: Efficient binary encoding of data structures
- **RPC interfaces**: Distributed service communication
- **Schema definitions**: Type-safe contracts between services

The schemas are located in ``src/python/fusionsc/serviceDefs/fusionsc/``.

Schema Types
------------

Structs
~~~~~~~

Data structures that define the layout of serialized data. Each struct has a
unique type ID and fields with assigned indices.

Interfaces
~~~~~~~~~~

Remote procedure call (RPC) interfaces that define service contracts. Each
method has a unique index and can return complex types.

API Documentation
-----------------

The documentation is automatically generated from the schema files. For each
schema, you'll find:

- Struct definitions with fields and types
- Interface definitions with methods and parameters
- Usage examples for C++ and Python
