Setting up FSC for your device
==============================

FSC supports high-level descriptions for magnetic configurations and geometries. However, the data backing these fields
or components are often restricted by team membership limitations, and can therefore usually not be shipped inside FSC.
Instead, in addition to connecting to the official databases (which requires on-premise network access), we support the
usage of `offline device files` that can be taken off-site and opened there. You can open a device file for usage in the
component / field resolution process, which will register it until your python interpreter / kernel is destroyed:

::

  import fusionsc as fsc
  fsc.resolve.importOfflineData('yourDeviceFile.fsc')

Wendelstein 7-X
---------------

On-site
~~~~~~~

When working on-site at Wendelstein 7-X and connected to the IPP network, you can use the on-premise coil and component data-
bases for resolution directly.

::

  import fsc.devices.w7x as w7x
  w7x.connectIPPSite()

Off-site
~~~~~~~~

Simply use an offline data file as outlined above.

Pre-calculating fields for Biot-Savart calculation
--------------------------------------------------

A generally expensive calculation is the Biot-Savart rule to obtain the magnetic field from the coil geometries. For select
devices, there is a high-level support to precompute the coil fields. These fields can then be saved and loaded later for
usage. When using precomputed coil fields, the offline data files are not required (unless required for other reasons, such
as geometry data or other field information).

Wendelstein 7-X
~~~~~~~~~~~~~~~

First, select the grid you want to calculate your coil fields over

::

  import fsc
  from fsc.devices import w7x
  
  grid = w7x.defaultGrid()
  grid.nR = 128
  grid.nZ = 128
  
Then, set up your calculation
::

  fsc.resolve.importOfflineData('w7x.fsc')
  # ... or ...
  w7x.connectIPPSite()
  
  tracer = fsc.tracer()
  
Finally, compute and save your coil fields:
::

  precomputedCoils = w7x.computeCoilFields(tracer.calculator, w7x.cadCoils())
  fsc.data.writeArchive(precomputedCoils, 'coils.fsc')

Later, you can then load the pre-computed coil fields and can use them as coil set arguments:
::

  import fsc
  from fsc.devices import w7x
  
  loadedCoils = fsc.data.loadArchive('coils.fsc')
  config = w7x.op12Standard(coils = loadedCoils) + w7x.controlCoils([200, -200], coils = loadedCoils)
  
  # Don-t forget to re-load the grid
  grid = w7x.defaultGrid()
  grid.nR = 128
  grid.nZ = 128