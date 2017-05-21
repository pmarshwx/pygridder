from distutils.core import setup

# Get Version Information
from pygridder import __version__

setup(
    name='pygridder',
    version=__version__,
    packages=['pygridder'],
    url='',
    license='MIT',
    author='Patrick Marsh',
    author_email='patrick.marsh@noaa.gov',
    description='A package to grid point, line, or polygon data on a regular grid'
)
