#pragma once

#include "common.h"
#include "jobs.h"

#include "kernels/device.h"

#include <fsc/vmec.capnp.h>
#include <fsc/jobs.capnp.h>

#include <kj/filesystem.h>

namespace fsc {

/** Create a VMEC service driver.
 *
 * @param dev: Compute device for the surface inversion calculations.
 * @param launcher: Job launcher to use for the VMEC code (post-processing is done using the system launcher).
 * @param config: Configuration for the VMEC driver.
 */
Own<VmecDriver::Server> createVmecDriver(Own<DeviceBase>&& dev, Own<JobLauncher>&& launcher, VmecConfig::Reader config);

/** VMEC input file generator.
 *
 * Generates the Fortran NAMELIST to be passed into VMEC.
 * @param request: The VMEC run request.
 * @param mgridPath: Path to the MGRID file required for free boundary runs.
 */
kj::String generateVmecInput(VmecRequest::Reader request, kj::PathPtr mgridPath);

/** MGRID file generator
 *
 * Writes a simple MGRID file from the computed field to the given destination path.
 */
Promise<void> writeMGridFile(kj::PathPtr path, ComputedField::Reader cField);

/** VMEC result reader
 *
 * Reads a wout.nc from VMEC and converts the contents into a response.
 * 
 * @warning This function can only read NetCDF4/HDF5 files. VMEC writes NetCDF-classic format.
 * Pre-process the output file with nccopy to convert it into NetCDF4.
 */
void interpretOutputFile(kj::PathPtr path, VmecResult::Builder out);

}
