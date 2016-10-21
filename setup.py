from setuptools import setup, Extension
import numpy as np

# source_suffix = 'pyx'
# has_cython = False
# try:
#     from Cython.Distutils import build_ext
#     has_cython = True
# except:
#     pass

# cmdclass={}
# if not has_cython:
source_suffix = 'c'
# else:
#    cmdclass = {'build_ext':build_ext}


costfunc = Extension('probfit.costfunc',
                     sources=['probfit/costfunc.' + source_suffix],
                     include_dirs=[np.get_include()],
                     extra_link_args=[])

pdf = Extension('probfit.pdf',
                sources=['probfit/pdf.' + source_suffix],
                include_dirs=[np.get_include()],
                extra_link_args=[])

libstat = Extension('probfit._libstat',
                    sources=['probfit/_libstat.' + source_suffix],
                    include_dirs=[np.get_include()],
                    extra_link_args=[])

funcutil = Extension('probfit.funcutil',
                     sources=['probfit/funcutil.' + source_suffix],
                     include_dirs=[np.get_include()],
                     extra_link_args=[])

functor = Extension('probfit.functor',
                    sources=['probfit/functor.' + source_suffix],
                    include_dirs=[np.get_include()],
                    extra_link_args=[])


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
    ext_modules=[costfunc, pdf, libstat, funcutil, functor],
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
