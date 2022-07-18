from skbuild import setup
from setuptools import find_packages

setup(
	name = 'fsc',
	version = '0.1',
	packages = find_packages(where = 'src/python'),
	package_dir = {'' : 'src/python'}
)

