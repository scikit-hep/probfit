from glob import glob
import os

from setuptools import setup
from setuptools.extension import Extension

import numpy as np

version = {}
with open('probfit/version.py') as fp:
    exec(fp.read(), version)

extensions = []
for source_file in glob('probfit/*.pyx'):
    fname, _ = os.path.splitext(os.path.basename(source_file))
    extensions.append(
        Extension('probfit.{0}'.format(fname),
                  sources=[source_file],
                  include_dirs=[np.get_include()])
    )


setup(
    version = version['__version__'],
    ext_modules=extensions,
)
