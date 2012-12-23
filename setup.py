from distutils.core import setup, Extension
import distutils.util
import subprocess
import numpy as np
source_suffix = 'pyx'
has_cython = False
try:
    from Cython.Distutils import build_ext
    has_cython = True
except:
    pass

cmdclass={}
if not has_cython:
    source_suffix = 'c'
else:
    cmdclass = {'build_ext':build_ext}


cdist_fit = Extension('probfit.cdist_fit',
                    sources = ['probfit/cdist_fit.'+source_suffix],
                    include_dirs= [np.get_include()],
                    extra_link_args = [])

cdist_func = Extension('probfit.cdist_func',
        sources = ['probfit/cdist_func.'+source_suffix],
        include_dirs= [np.get_include()],
        extra_link_args = [])

common = Extension('probfit.common',
        sources = ['probfit/common.'+source_suffix],
        include_dirs= [np.get_include()],
        extra_link_args = [])

toy = Extension('probfit.toy',
        sources = ['probfit/toy.'+source_suffix],
        include_dirs= [np.get_include()],
        extra_link_args = [])

setup (
       cmdclass=cmdclass,
       name = 'probfit',
       version = '1.0.0',
       description = 'Distribution Fitting/Regression Library',
       author='Piti Ongmongkolkul',
       author_email='piti118@gmail.com',
       url='https://github.com/piti118/dist_fit',
       package_dir = {'probfit': 'probfit'},
       packages = ['probfit'],
       ext_modules = [cdist_fit,cdist_func,common, toy],
       requires=['numpy','iminuit']
       )