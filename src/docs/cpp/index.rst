C++ Library Documentation
=========================

This section documents the C++ core library of FusionSC. The C++ API provides
the high-performance computation kernels and services that the Python bindings
wrap.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules:

   magnetics
   fieldline-tracing
   geometry
   data
   vmec
   services

.. toctree::
   :maxdepth: 1
   :caption: Generated API Reference:

   genapi

Quick Start
------------

The C++ library is built as part of the main CMake build. See
:doc:`/installation` for build instructions.

The main library target is ``fsc``, which links against all required
dependencies. Header files are organized by module under ``src/c++/fsc/``.
