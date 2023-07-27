Installation
============

**Note: Currently we are in pre-release for version 2. Please install the newest version with

::

  pip install --pre fusionsc

FSC can be installed from PyPi. Binaries are provided for Windows, while for Linux generally a source-build is run.
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