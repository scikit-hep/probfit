from glob import glob
import os
from setuptools import setup
from setuptools.extension import Extension

import numpy as np

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    print('Cython is not available; using pre-generated C files')
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'
extensions = []
for source_file in glob('probfit/*' + ext):
    fname, _ = os.path.splitext(os.path.basename(source_file))
    extensions.append(
        Extension('probfit.{0}'.format(fname),
                  sources=['probfit/{0}{1}'.format(fname, ext)],
                  include_dirs=[np.get_include()])
    )

if USE_CYTHON:
    extensions = cythonize(extensions)


def get_version():
    version = {}
    with open('probfit/info.py') as fp:
        exec(fp.read(), version)
    return version['__version__']


__version__ = get_version()

setup(
    name='probfit',
    version=__version__,
    description='Distribution Fitting/Regression Library',
    long_description=''.join(open('README.rst').readlines()[4:]),
    author='Piti Ongmongkolkul',
    author_email='piti118@gmail.com',
    url='https://github.com/iminuit/probfit',
    package_dir={'probfit': 'probfit'},
    packages=['probfit'],
    ext_modules=extensions,
    install_requires=[
        'setuptools',
        'numpy',
        'iminuit',
    ],
    classifiers=[
        "Programming Language :: Python",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License'
    ],
)
