Data Management
===============

The data module provides HDF5-backed storage, Cap'n'Proto serialization,
and archive file handling.

Overview
--------

FusionSC uses a statically typed, cross-version-compatible, high-speed binary
message format for data storage and handling. It provides three persistence
mechanisms:

- **Archive files**: Store an entire tree of a root message and its dependencies
  in a single immutable file. Optimized for fast writing and extremely fast reading.
- **Structured IO**: For import and export to other programs, supports JSON, YAML,
  CBOR, and BSON formats.
- **Warehouses**: Mutable object stores backed by a local SQLite database,
  supporting multi-user access, compact storage, and backup facilities.

API Reference
-------------

.. doxygenfile:: data.h
   :project: fsc

.. doxygenfile:: blob-store.h
   :project: fsc

.. doxygenfile:: store.h
   :project: fsc

.. doxygenfile:: db.h
   :project: fsc
