#!/usr/bin/env python

from skbuild import setup

setup (
        name    = 'pysimanneal',
        version = '0.2.1',
        cmake_with_sdist = True,
        packages = ['pysimanneal'],
        zip_safe = False
        )