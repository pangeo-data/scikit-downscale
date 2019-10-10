#!/usr/bin/env python

from setuptools import setup, find_packages

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
]

setup(name='xsd',
      author='Joe Hamman',
      author_email='jhamman@ucar.edu',
      license="Apache",
      classifiers=CLASSIFIERS,
      description='Statistical downscaling and postprocessing models for climate and weather model simulations.',
      python_requires=">=3.6",
      install_requires=["xarray >= 0.10", "scikit-learn >= 0.21"],
      tests_require=["pytest >= 2.7.1"],
      url='https://github.com/jhamman/xsd',
      packages=find_packages(),
      )
