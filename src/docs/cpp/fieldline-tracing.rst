Field Line Tracing
==================

The field line tracing module computes magnetic field line trajectories,
Poincare plots, and connection lengths using reversible mapping techniques.

Overview
--------

The field line tracer supports:

- Computation of **reversible field line mappings** and usage of these mappings
  to accelerate all field line calculations
- Calculation of **Poincare plots** with forward- and backward-connection-lengths
  (at no extra cost)
- **Convective-diffusive** and **doubly-diffusive** field line tracing
- Calculation of **connection lengths**, strike positions, and geometry tags
  (module numbers, finger IDs, etc.) on impact points

API Reference
-------------

.. doxygenfile:: flt.h
   :project: fsc

.. doxygenfile:: fieldline-mapping.h
   :project: fsc
