Magnetics
=========

The magnetics module provides magnetic field computation capabilities including
Biot-Savart coil fields, dipole fields, and field interpolation.

Overview
--------

The magnetic field system in FusionSC supports several types of field sources:

- **Coil filament fields** based on the Biot-Savart law (with finite-wire regularization)
- **Dipole fields** from magnetized spheres (constant field within sphere)
- **Interpolation** of 3D fields corresponding to 2D equilibria (e.g. EFIT files)
- **Cross-interpolation** between different grids
- **Geometric operations** (rotation, shift) applied to fields

Fields are resolved through a ``FieldResolver`` service that expands high-level
field specifications into low-level computational descriptions.

API Reference
-------------

.. doxygenfile:: magnetics.h
   :project: fsc
