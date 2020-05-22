#!/usr/bin/env python

from skbuild import setup

setup (
        name    = 'pysimanneal',
        version = '0.2.1',
        cmake_with_sdist = True,
        packages = ['pysimanneal'],
        zip_safe = False
        )

##!/usr/bin/env python
#
#'''
#setup.py file for SWIG example
#'''
#
#from distutils.core import setup, Extension
#import os
#os.environ["CC"] = "g++"
#os.environ["CXX"] = "g++"
#
#simanneal_module = Extension('_simanneal',
#                           sources=['simanneal_wrap.cxx', 'simanneal.cc', 
#                               'global.cc'],
#                           )
#
#setup (name = 'simanneal',
#       version = '0.1',
#       author      = 'Samuel Ng',
#       description = '''Python wrapper for SimAnneal ground state finder.''',
#       ext_modules = [simanneal_module],
#       py_modules = [],
#       )
