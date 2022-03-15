#! /usr/bin/env python

from setuptools import setup


setup(name='landing',
      install_requires=['torch', 'geoopt', 'scipy'],
      packages=['landing'],
      version='0.0'
      )