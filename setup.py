from skbuild import setup
from setuptools import find_packages

setup(
	name = 'fsc',
	version = '0.1',
	packages = ['fsc', 'fsc.devices', 'fsc.devices.w7x'],
	package_dir = {"" : "src/python"},
	cmake_install_target = "fsc-install-skbuild",
	entry_points = {
		'console_scripts': [
			'w7x-download = fsc.devices.w7x.scripts:download'
		]
	},
	
	install_requires = [
		"typing-extensions"
	]
)