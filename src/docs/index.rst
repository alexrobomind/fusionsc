.. FSC documentation master file, created by
   sphinx-quickstart on Mon Aug 22 13:43:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FusionSC's documentation!
====================================

What is FusionSC?
-----------------

Started as an effort to automate my personal workflow, FusionSC is a library intended to conveniently expose common
operations required when analyzing the structure of magnetic confinement fusion devices, including the structure of
the magnetic field as well as the machine geometry. 

**Currently fusionsc is in alpha release for version 2. This version includes minor interface overhauls which will
be kept.** Most importantly, `defaultGrid` and `defaultGeometryGrid` are now **functions for all devices**. Beyond
that, version 2 has full asyncio integration, which was a large item on the user-facing side.

We do not expect any interface changes beyond version 2 any time soon. However, please keep in mind that the release
version 2 will possibly implement additional feedback.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   setup
   howto/index
   architecture
   dataspec
   license
  
   reference



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
