from glob import glob
import logging
import os
from setuptools import setup
from setuptools.extension import Extension

logging.basicConfig()
log = logging.getLogger('probfit')


class NumpyExtension(Extension):
    """C Extension that implicitly uses numpy.

    This class can be useful because it defers the importing of the numpy
    module until the include_dirs property is accessed. For setup.py commands
    like egg_info, include_dirs is not queried, meaning numpy isn't required
    and shouldn't be a dependency.

    Taken from the "NumPy extensions and setup_requires" thread in the
    distutils mailing list.
    """
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)

        self._include_dirs = self.include_dirs
        del self.include_dirs

    @property
    def include_dirs(self):
        from numpy import get_include

        return self._include_dirs + [get_include()]


def get_extensions():
    extensions = []
    try:
        from Cython.Build import cythonize

        use_cython = True
    except ImportError:
        log.warning('Cython is not available; using pre-generated C files')
        use_cython = False

    ext = '.pyx' if use_cython else '.c'
    for source_file in glob('probfit/*' + ext):
        log.info('Adding extension for file {0!r}'.format(source_file))
        fname, _ = os.path.splitext(os.path.basename(source_file))
        extensions.append(
            NumpyExtension('probfit.{0}'.format(fname),
                           sources=['probfit/{0}{1}'.format(fname, ext)])
        )

    if not extensions:
        # If we get here, the user has numpy installed but there are no .c
        # files to build, so they must be generated which requires Cython
        log.error('Could not build extensions; you must install Cython')

    if use_cython:
        extensions = cythonize(extensions)

    return extensions


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
    ext_modules=get_extensions(),
    install_requires=[
        'setuptools',
        'numpy',
        'iminuit',
        'matplotlib'
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
