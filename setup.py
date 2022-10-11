#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 2022

@author: Géraldine Zenhäusern
"""
from setuptools import setup, find_packages


setup(name='polarisation_package',
      version='1.0',
      description='Polarisation analysis of seismic data to determine the back azimuth',
      url='https://github.com/gzenhaeusern/xxx',
      author='Géraldine Zenhäusern',
      author_email='geraldine.zenhaeusern@erdw.ethz.ch',
      license='xxx',
      packages=find_packages(),
      install_requires=['obspy', 'numpy', 'matplotlib',
                        'scipy', 'argparse', 'seaborn'])
