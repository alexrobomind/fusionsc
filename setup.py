from skbuild import setup
from setuptools import find_packages

setup(
	name = 'fsc',
	version = '0.1',
	packages = ['fsc'],
	package_dir = {"" : "src/python"},
	cmake_install_target = "fsc-install-skbuild",
	entry_points = {
		'console_scripts': [
			'w7x-downloa = fsc.w7x.scripts:download'
		]
)