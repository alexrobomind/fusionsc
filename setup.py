from skbuild import setup
from setuptools import find_packages

setup(
	name = 'fusionsc',
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
		"typing-extensions",
		"numpy>=1.19.5 ; python_version < '3.10'",
		"numpy>=1.23.4 ; python_version >= '3.10'",
	]
)