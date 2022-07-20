from fsc.native.devices.w7x import (
	defaultGrid, webserviceCoilsDB, webserviceComponentsDB
)

from fsc.native import W7XCoilSet, newCache, download
from fsc.asnc import asyncFunction, eager
from fsc.flt import Config
from fsc.resolve import fieldResolvers, geometryResolvers, resolveField

from fsc import native

from typing import Union

# Databases

def connectCoilsDB(address: str):
	global fieldResolvers
	
	coilsDB = webserviceCoilsDB(address)
	coilsDBResolver = cppw7x.coilsDBResolver(coilsDB)
	fieldResolvers.append(coilsDBResolver)
	
	return coilsDB

def connectComponentsDB(address: str):
	global geometryResolvers
	
	componentsDB = webserviceComponentsDB(address)
	componentsDBResolver = cppw7x.componentsDBResolver(componentsDB)
	geometryResolvers.append(componentsDBResolver)
	
	return componentsDB

# The W7XCoilSet type defaults to the W7-X coils 160 ... 230
cadCoilPack = W7XCoilSet.newMessage()

coil_conventions = ['coilsdb', '1-AA-R0004.5', 'archive'];

def process_coil_convention(convention):
	assert convention in coil_conventions,	'Invalid coil convention {}, must be one of {}'.format(convention, coil_conventions);
	
	if convention == 'archive':
		return '1-AA-R0004.5';
	
	return convention;

@eager
async def preheat(tracer, grid = defaultGrid(), coilPack: Union[W7XCoilSet.Builder, W7XCoilSet.Reader] = cadCoilPack, check: bool = False):
	"""
	Starts a pre-computation for the fields created by the given coil pack
	"""
	# Compute the fields that we need to pre-calculate for
	fields = native.devices.w7x.preheatFields(coilPack)
		
	for field in fields:
		field = await resolveField(field)
		print(field)
		computed = (await tracer.calculator.compute(field, grid)).computedField
		
		cache = newCache(field, computed)
		fieldResolvers.append(cache)
		
		if(check):
			await download(computed.data)

def cadCoils(i_12345 = [0, 0, 0, 0, 0], i_ab = [0, 0], i_trim = [0, 0, 0, 0, 0], i_cc = [0, 0], convention = '1-AA-R0004.5'):
	assert len(i_12345) == 5;
	assert len(i_ab) == 2;
	assert len(i_trim) == 5;
	assert len(i_cc) in [2, 5, 10];
	
	config = flt.Config()
	
	# Declare field as W7-X 'Coils and currents' field-type and 
	# set coils
	cnc = config.field.initW7xMagneticConfig().initCoilsAndCurrents()
	cnc.coils = cadCoilPack
	
	convention = process_coil_convention(convention)
	if convention == 'coilsdb':
		cnc.coils.invertMainCoils = False
	else:
		cnc.coils.invertMainCoils = True
	
	cnc.nonplanar = i_12345
	cnc.planar = i_ab
	cnc.trim = i_trim
	cnc.control = i_cc
	
	return config