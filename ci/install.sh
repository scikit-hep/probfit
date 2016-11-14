#!/bin/bash

# Exit after any command returns a non-zero code, and print line numbers as
# they are run
set -ev

if [[ "$BUILD_PACKAGE" == "TRUE" ]]; then
  pip install cython
fi

pip install -r probfit.egg-info/requires.txt

if [[ "$INSTALL_OPT_DEPS" == "TRUE" ]]; then
  # pip install matplotlib
  echo 'no optional dependencies to install'
fi

if [[ "$RUN_TESTS" == "TRUE" ]]; then
  pip install pytest pytest-cov pytest-mpl
fi

if [[ "$RUN_LINTERS" == "TRUE" ]]; then
  pip install flake8 pylint
fi

if [[ "$BUILD_DOCS" == TRUE ]]; then
  pip install ipython sphinx sphinx_rtd_theme
fi
