# -*- coding: utf-8 -*-
import os
from glob import glob

import numpy as np
from setuptools import setup
from setuptools.extension import Extension

version = {}
with open("probfit/version.py") as fp:
    exec(fp.read(), version)

extensions = []
for source_file in glob("probfit/*.pyx"):
    fname, _ = os.path.splitext(os.path.basename(source_file))
    extensions.append(
        Extension(
            "probfit.{}".format(fname),
            sources=[source_file],
            include_dirs=[np.get_include()],
        )
    )


setup(
    version=version["__version__"],
    ext_modules=extensions,
)
