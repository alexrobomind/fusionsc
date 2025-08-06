Overview & Features
===================

The primary task of FusionSC is to assist with the calculation of 3D MHD equilibria and
to assist with the analysis of 3D magnetic fields and geometries. For this purpose, it
combines a suite of code drivers and fast parallel computation kernels commoly used in
magnetic confinement fusion calculations.

Built-in physics
----------------

FusionSC comes with a series of fast built-in physics modules intended to handle the
basic aspects of 3D MCF physics:

- A flexible 3D magnetic field computation engine that supports various operations:
  - Coil filament fields based on Biot-Savart law (with finite-wire regularization)
  - Dipole fields from magnetized spheres (constant field within sphere)
  - Interpolation of 3D fields corresponding to 2D equilibria (e.g. EFIT files)
  - Cross-interpolation between different grids
  - Geometric operations (rotation, shift) applied to the fields

- An advanced field line tracer module, including the following capabilities:

  - Computation of reversible field line mappings and usage of these mappings
    to accelerate all field line calculations
  - Calculation of Poincare-plots with forward- and backward-connection-lengths
    (at no extra cost)
  - Convective-diffusive and doubly-diffusive field line tracing
  - Calculation of connection lengths, strike positions, and geometry tags (
    module numbers, finger IDs, etc.) on impact points

- A virtual camera system to calculate heat-loads on 3D geometries based on 
  3D strike points

External code interfaces
------------------------

Beyond the internally provided physics modules, FusionSC also can launch HPC
codes, either by directly launching system processes (shell-style) or via mpiexec
(support for SLURM is in development).

- Execution of VMEC runs
- Currently work in progress: Execution of HINT runs

Machine geometry handling
-------------------------

Geometries and fields can be created from high-level field specifications, which
are then resolved into coil geometries and mesh definitions using fast lookup
tables. The geometry subsystem provides support for the easy specification of
parts:

- Flexible geometry specification including meshes, transformations, label
  annotations, and toroidal extrusion of 2D part geometries
- Geometry visualization / VTK export using PyVista
- 3D indexing of geometry for fast intersection calculations
- Ray-cast operations on indexed geometries

Remote & asynchronous computation
---------------------------------

While shipping with a fully functional local backend, FusionSC has first-class
support for remote execution. Calculation (and some data handling) requests can
be delegated to a remote instance of the user's choice. The connection is es-
tablished through a fast asynchronous RPC protocol based on HTTP & websocket.

FusionSC is integrated with asyncio to allow the simultaneous scheduling and
execution of a large number of different tasks.

Data storage and distribution
-----------------------------

FusionSC uses a statically typed cross-version-compatible high-speed binary
message format for data storage and handling to reduce the overhead of the non-
physics portions of the code. It provide three persistence mechanisms for these
data objects:

- *Archive files*: Archive files store an entire tree of a root message and
  its dependencies in a single immutable file. These files are optimized for
  fast writing and extremely fast reading. When downloading remote data into
  archive files, they are streamed directly to their respective file portions
  without in-memory assembly (reducing memory overhead).
  
  When opening archive files, their contents are directly mapped into the
  process's memory space, eliminating read overhead for unused portions of the
  contained data and allowing the OS to easily reclaim memory pages when
  facing memory pressure.

- *Structured IO*: For import & export to other programs, as well as human IO,
  objects may be read and written in a variety of self-describing nested formats.
  The structured IO subsystem currently supports JSON and YAML for textual I/O
  as well as CBOR and BSON for binary input & output.

- *Warehouses*: Warehouses are mutable object stores backed by a local
  database (sqlite). While archives can only be written once, warehouses
  expose mutable folder- and file-type classes that can be used to store and
  organize extremely large data collections.
  
  Trading raw speed for scalability, warehouses are designed as an endpoint
  to share large databases, incorporating the following features:
  
  - *Multi-user access*: Warehouses can be safely read and modified
    simultaneously by multiple threads and/or processes. Additionally, they
    can be served for remote access.
  - *Compact storage*: Objects containing identical data will always share
    their underlying storage, eliminating data duplication (archive files do
    this as well). In addition, all stored data are compressed using deflate.
  - *Protection against corruption*: Being based on SQLite, warehouses inherit
    its strong protection against system crashes, allowing recovery to a
    consistent state.
  - *Backup facilities*: The underlying database can be backed up during
    during operation to guard against fatal storage loss.

Beyond its built-in static schema types, FusionSC also supports the storage
of a large number of common python types:

- Primitive types (:code:`bytes`, :code:`str`, :code:`int`) with zero-copy
  loading for :code:`bytes`
- Numpy arrays, including optimized representations for numpy arrays holding
  native FusionSC objects
- Sequences (e.g. :code:`list`, :code:`tuple`)
- Mapping (e.g. :code:`dict`)

Furthermore, FusionSC can also use optionally leverage pickle to store and
load a larger variety of objects. Due to the security concerns surrounding the
use of pickle (loading pickle can involve arbitrary code execution), this
feature is disabled by default but is optionally available.