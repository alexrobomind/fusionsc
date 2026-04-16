VMEC Interface
==============

Interface to the VMEC 3D MHD equilibrium code.

Overview
--------

FusionSC can launch and interact with VMEC runs, either by directly launching
system processes or via ``mpiexec``. The VMEC interface handles configuration,
execution, and result extraction.

API Reference
-------------

.. doxygenfile:: vmec.h
   :project: fsc
