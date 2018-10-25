# Makefile with some convenient quick ways to do common things

PROJECT = probfit
CYTHON ?= cython

PYX_FILES := $(wildcard probfit/*.pyx)

.PHONY: help clean build test coverage doc doc-show code-analysis flake8 pylint

help:
	@echo ''
	@echo ' probfit available make targets:'
	@echo ''
	@echo '     help             Print this help message (the default)'
	@echo ''
	@echo '     clean            Remove generated files'
	@echo '     build            Build inplace'
	@echo '     test             Run tests'
	@echo '     coverage         Run tests and write coverage report'
	@echo '     doc              Run Sphinx to generate HTML docs'
	@echo '     doc-show         Open local HTML docs in browser'
	@echo ''
	@echo '     code-analysis    Run code analysis (flake8 and pylint)'
	@echo '     flake8           Run code analysis (flake8)'
	@echo '     pylint           Run code analysis (pylint)'
	@echo ''
	@echo ' Note that most things are done via `python setup.py`, we only use'
	@echo ' make for things that are not trivial to execute via `setup.py`.'
	@echo ''
	@echo ' Common `setup.py` commands:'
	@echo ''
	@echo '     python setup.py --help-commands'
	@echo '     python setup.py install'
	@echo '     python setup.py develop'
	@echo ''
	@echo ' More info:'
	@echo ''
	@echo ' * probfit code: https://github.com/scikit-hep/probfit'
	@echo ' * probfit docs: https://probfit.readthedocs.io'
	@echo ''

clean:
	rm -rf build htmlcov doc/_build
	$(MAKE) -C doc clean
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find . -name "*.c" -exec rm {} \;
	find . -name __pycache__ | xargs rm -fr

build: $(PYX_FILES)
	python setup.py build_ext --inplace

test: build
	python -m pytest -v
	python -m pytest -v --mpl tests/test_plotting.py

coverage: build
	python -m pytest -v $(PROJECT) --cov $(PROJECT) --cov-report html --cov-report term-missing --cov-report xml

doc: build
	$(MAKE) -C doc html

doc-show: doc
	open doc/_build/html/index.html

code-analysis: flake8 pylint

flake8:
	flake8 --max-line-length=90 $(PROJECT) | grep -v __init__ | grep -v external

# TODO: once the errors are fixed, remove the -E option and tackle the warnings
pylint: build
	pylint -E $(PROJECT)/ -d E1103,E0611,E1101 \
	       --ignore="" -f colorized \
	       --msg-template='{C}: {path}:{line}:{column}: {msg} ({symbol})'
