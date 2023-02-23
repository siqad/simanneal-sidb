#!/usr/bin/env python

from distutils.command.build_ext import build_ext
from skbuild import setup


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup (
    name    = 'pysimanneal',
    version = '0.2.1',
    cmake_with_sdist = True,
    packages = ['pysimanneal'],
    cmake_languages = ('C', 'CXX', 'CUDA'),
    cmdclass={'build_ext': custom_build_ext},
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
