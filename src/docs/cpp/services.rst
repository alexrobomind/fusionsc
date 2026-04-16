Services
========

The services module implements the RPC service layer for remote computation
and the local in-process server.

Overview
--------

FusionSC has first-class support for remote execution. Calculation and data
handling requests can be delegated to a remote instance. The connection is
established through a fast asynchronous RPC protocol based on HTTP and WebSocket,
built on Cap'n'Proto.

API Reference
-------------

.. doxygenfile:: services.h
   :project: fsc

.. doxygenfile:: local.h
   :project: fsc

.. doxygenfile:: in-process-server.h
   :project: fsc

.. doxygenfile:: offline.h
   :project: fsc
