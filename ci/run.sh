#!/bin/bash

# Exit after any command returns a non-zero code, and print line numbers as
# they are run
set -ev

if [[ "$BUILD_PACKAGE" != "TRUE" ]]; then
  # Building the package with sdist _should_ build the C extensions
  make build
else
  python setup.py sdist
  cd dist
  virtualenv venv
  source venv/bin/activate
  pip install *.tar.gz
  python -c 'import probfit'
  deactivate
  cd -
fi

if [[ "$RUN_LINTERS" == "TRUE" ]]; then
  make flake8
  make pylint
fi

if [[ "$RUN_TESTS" == "TRUE" ]]; then
  make test
fi

if [[ "$BUILD_DOCS" == TRUE ]]; then
  make doc
fi
