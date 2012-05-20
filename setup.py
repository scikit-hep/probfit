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


cdist_fit = Extension('dist_fit.cdist_fit',
                    sources = ['dist_fit/cdist_fit.'+source_suffix],
                    include_dirs= [np.get_include()],
                    extra_link_args = [])

cdist_func = Extension('dist_fit.cdist_func',
        sources = ['dist_fit/cdist_func.'+source_suffix],
        include_dirs= [np.get_include()],
        extra_link_args = [])

common = Extension('dist_fit.common',
        sources = ['dist_fit/common.'+source_suffix],
        include_dirs= [np.get_include()],
        extra_link_args = [])

setup (
       cmdclass=cmdclass,
       name = 'dist_fit',
       version = '1.00',
       description = 'Distribution Fitting/Regression Library',
       author='Piti Ongmongkolkul',
       author_email='piti118@gmail.com',
       url='https://github.com/piti118/dist_fit',
       package_dir = {'dist_fit': 'dist_fit'},
       packages = ['dist_fit'],
       ext_modules = [cdist_fit,cdist_func,common], requires=['numpy']
       )