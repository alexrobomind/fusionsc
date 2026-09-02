:: _roadmap:

Roadmap
~~~~~~~

Currently, FusionSC exposes a core set of service provided locally. However, a key goal was always to provide remote
call functionality and handle supercomputer-based codes, which are more cumbersome to use manually. To this end, the
following features are intended to be added into the code rather sooner than later:

* Version 1.0 (released):

  * Data transfer, sharing, and archiving
  * Field line tracing, field line diffusion, and heat flux inversion services
  * Multi-threaded CPU worker implementation
	
* Version 1.1 (released):
  Remote connection facilities

  * Connection to remote worker instances
  * Stand-alone and in-python server
  * SSH connections
	
* Version 1.2:
  Infrastructure for extended mode
  
  * Spack build
  * System launcher
  * Field line following
  * Pickling support
	  
* Version 1.3:

  * ObjectDB (database with mutable folder strucure to store result data)
  * VMEC support
  
    * NetCDF support library in extended mode build
    * VMEC input generators and output parsers
	
  * SLURM launcher

* Version 1.4:
  HINT support