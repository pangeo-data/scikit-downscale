#!/usr/bin/env python

from setuptools import find_packages, setup

NAME = 'scikit-downscale'
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering',
]
PACKAGES = find_packages(exclude=['*test*'])
print(f'installing {PACKAGES}')

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name=NAME,
    author='Joe Hamman',
    author_email='jhamman@ucar.edu',
    license='Apache',
    classifiers=CLASSIFIERS,
    description='Statistical downscaling and postprocessing models for climate and weather model simulations.',
    python_requires='>=3.6',
    install_requires=install_requires,
    tests_require=['pytest >= 2.7.1'],
    url='https://github.com/jhamman/scikit-downscale',
    packages=PACKAGES,
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0'],
)
