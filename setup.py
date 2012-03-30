from distutils.core import setup, Extension
import distutils.util
import subprocess
import numpy as np

module = Extension('dist_fit.cdist_fit',
                    sources = ['src/cdist_fit.c'],
                    include_dirs= [np.get_include()],
                    extra_link_args = [])

setup (name = 'dist_fit',
       version = '1.00',
       description = 'Distribution fitting/Regression library',
       author='Piti Ongmongkolkul',
       author_email='piti118@gmail.com',
       url='https://github.com/piti118/dist_fit',
       package_dir = {'dist_fit': 'src'},
       packages = ['dist_fit'],
       ext_modules = [module]
       )