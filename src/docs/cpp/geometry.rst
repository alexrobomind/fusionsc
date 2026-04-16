Geometry
========

The geometry module handles 3D mesh processing, PLY import/export,
and spatial indexing for intersection calculations.

Overview
--------

FusionSC offers a high-level geometry representation based on meshes,
declarative operations (2D to 3D extrusions, geometric operations on other
geometries), and labeling support.

Geometry processing is primarily handled through the ``GeometryLib`` service
interface. Geometries are fully represented using Cap'n'Proto types.

API Reference
-------------

.. doxygenfile:: geometry.h
   :project: fsc
