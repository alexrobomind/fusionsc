"""Helpers to describe and load HINT equilibria"""

from . import service, magnetics, wrappers, data

from .asnc import asyncFunction
from .data import publish

from ._api_markers import untested

import netCDF4
import numpy as np
import warnings
import scipy.io

class HintEquilibrium(wrappers.structWrapper(service.HintEquilibrium)):	
	def asField(self):
		"""Extracts a magnetic configuration from the equilibrium"""
		result = magnetics.MagneticConfig()
		
		comp = result.data.initComputedField()
		comp.grid = self.data.grid
		comp.data = self.data.field
		
		return result
	

def loadNetcdfFile(filename):	
	"""Creates a HINT equilibrium from the given netCDF HINT file"""
	with netCDF4.Dataset(filename) as f:
		m_tor = f.variables['mtor'][...].item()
		rminb = f.variables['rminb'][...].item()
		rmaxb = f.variables['rmaxb'][...].item()
		zminb = f.variables['zminb'][...].item()
		zmaxb = f.variables['zmaxb'][...].item()
		
		if 'Bv_phi' in f.variables:
			warnings.warn('Loading vacuum file, v and p will be set to zero')
		
			b_field = np.stack(
				[
					f.variables['Bv_phi'][...],
					f.variables['Bv_Z'][...],
					f.variables['Bv_R'][...]
				],
				axis = -1
			)
			
			v_field = np.zeros(b_field   .shape, dtype = np.float32)
			p_field = np.zeros(b_field[0].shape, dtype = np.float32)
			
		else:
			b_field = np.stack(
				[
					f.variables['B_phi'][...],
					f.variables['B_Z'][...],
					f.variables['B_R'][...]
				],
				axis = -1
			)
			
			v_field = np.stack(
				[
					f.variables['v_phi'][...],
					f.variables['v_Z'][...],
					f.variables['v_R'][...]
				],
				axis = -1
			)
			
			p_field = f.variables['P'][...]
		
		return _buildHint(
			b_field, p_field, v_field,
			rminb, rmaxb, zminb, zmaxb,
			m_tor,
			*(p_field.shape)
		)

@untested
def loadFortranSnapfile(filename, big_endian = True, real_bytes = 8):
	"""Creates a HINT equilibrium from the given HINT FORTRAN file."""
	# Define types
	if big_endian:
		int_type = '>i4'
		real_type = '>f{}'.format(real_bytes)
		header_type = '>u4'
	else:
		int_type = 'i4'
		real_type = 'f{}'.format(real_bytes)
		header_type = 'u4'

	# Read data from fortran file
	f = scipy.io.FortranFile(filename, 'r', header_dtype = header_type)

	kstep = f.read_ints(int_type)
	time  = f.read_reals(real_type)
	n_r, n_z, n_p, m_tor = f.read_ints(int_type)
	r_min, z_min, r_max, z_max = f.read_reals(real_type)

	b_field = f.read_reals(real_type)
	b_field = b_field.reshape([n_p, n_z, n_r, 3])

	v_field = f.read_reals(real_type)
	v_field = v_field.reshape([n_p, n_z, n_r, 3])

	p_field = f.read_reals(real_type);
	p_field = p_field.reshape([n_p, n_z, n_r])
	
	# Convert from r, phi, z to phi, z, r component order
	b_field = b_field[..., [1, 2, 0]]
	v_field = v_field[..., [1, 2, 0]]

	f.close()
	
	return _buildHint(
		b_field, p_field, v_field,
		r_min, r_max, z_min, z_max,
		m_tor,
		n_p, n_z, n_r
	);

def _buildHint(b_field, p_field, v_field, r_min, r_max, z_min, z_max, m_tor, n_phi, n_z, n_r):
	assert b_field.shape == (n_phi, n_z, n_r, 3)
	assert p_field.shape == (n_phi, n_z, n_r)
	
	# Pre-process fields
	mu_0 = 4e-7 * np.pi
	p_field = np.asarray(p_field) * (1 / mu_0)
		
	equi = service.HintEquilibrium.newMessage()
	
	grid = equi.grid
	grid.rMin = r_min
	grid.rMax = r_max
	grid.zMin = z_min
	grid.zMax = z_max
	grid.nSym = m_tor
	grid.nR = n_r
	grid.nZ = n_z
	grid.nPhi = n_phi
	
	equi.field = data.publish(service.Float64Tensor.newMessage(b_field))
	equi.pressure = data.publish(service.Float64Tensor.newMessage(p_field))
	equi.velocity = data.publish(service.Float64Tensor.newMessage(v_field))
	
	return HintEquilibrium(equi)