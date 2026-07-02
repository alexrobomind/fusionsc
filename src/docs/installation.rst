Installation
============

FusionSC can be installed from PyPi. Binaries are provided for Windows, while for Linux generally a source-build is run.
To install, simply run

::

  pip install fusionsc

Alternatively, you can download fsc from the repository and install a chosen version from source:

::

  git clone https://jugit.fz-juelich.de/a.knieps/fsc.git
  cd fsc
  pip install .

Verifying the installation
--------------------------

To verify the installation, validate that you can import the `fusionsc` python module

::

  >>> import fusionsc as fsc


Installing external dependencies
--------------------------------

FusionSC itself does not require external dependencies to function, however some of its code drivers need external
dependencies.

* The VMEC driver requires **VMEC** and the **NetCDF binary programs** (specificially the *nccopy* tool to convert
  VMEC's NetCDF classic output files into HDF5-based NetCDF 4 libraries, as FusionSC can not read classic files). Both
  of these have to be present **on the server**.
  
  At default setting, VMEC is expected to be on the PATH as `xvmec2000`, but any other command can be set in the VMEC driver's
  configuration.