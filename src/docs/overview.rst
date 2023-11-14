Overview & Features
===================

The primary task of FusionSC is to assist with the calculation of 3D MHD equilibria and
to assist with the analysis of 3D magnetic fields and geometries. For this purpose, it
combines a suite of code drivers and fast parallel computation kernels commoly used in
magnetic confinement fusion calculations.

Built-in physics
----------------

FusionSC comes with a series of fast built-in physics modules:

- Biot-Savart calculations to obtain 3D vacuum fields from coil descriptions
- Interpolation of axisymmetric 3D fields from 2D equilibria (incl. EFIT files)
- An advanced field line tracer module, including the following capabilities:

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
codes, either by directly launching system processes (shell-style), via mpiexec,
or using slurm.

- Execution of VMEC runs
- HINT support is planned

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
be delegated to a remote instance of the user's choice. FusionSC is integrated
with asyncio to allow the simultaneous scheduling and execution of a large
number of different tasks.

Data storage and distribution
-----------------------------

FusionSC uses a statically typed cross-version-compatible high-speed binary
message format for data storage and handling to reduce the overhead of the non-
physics portions of the code. It provide two persistence mechanisms for these
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

- *Warehouses*: Warehouses are mutable object stores backed by a local
  database (sqlite). While archives can only be written once, warehouses
  expose mutable folder- and file-type classes that can be used to store and
  organize extremely large data collections.
  
  Trading raw speed for scalability, warehouses are designed as an endpoint
  to share large databases, incorporating the following features:
    - Multi-user access (multiple processes and remote network-based access)
	- Compressed data storage
	- Protection against system crashes
	- Backup facilities

Beyond its built-in static schema types, FusionSC also supports the storage
of a large number of common python types:

- Primitive types (bytes, str, int) with zero-copy loading for bytes
- Numpy arrays (with the exception of struct types), including special
  optimized representations for numpy arrays holding native FusionSC
  objects
- Sequences (e.g. list, tuple)
- Mapping (e.g. dict)

Furthermore, FusionSC can also use optionally leverage pickle to store and
load a larger variety of objects. Due to the security concerns surrounding the
use of pickle (loading pickle can involve arbitrary code execution), this
feature is disabled by default but is optionally available.