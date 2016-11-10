.. _development:

Development
===========

Contributions to probfit are welcome. You should fork the repository,
create a branch in your fork, and then `open a pull request 
<https://github.com/iminuit/probfit/pulls>`_.

Developing probfit requires a few dependencies, in addition to those required
for users. The following commands should create a suitable environment,
assuming you've cloned your fork of probfit and are in the repository root.
(You may wish to work inside a virtual environment to isolate these packages
from your system install.)

.. code-block:: shell

    $ pip install cython pytest pytest-mpl pylint flake8 sphinx sphinx_rtd_theme
    $ make build

Installing `Cython <http://cython.org/>`_ will allow you to build the C
extensions. `Pylint <https://www.pylint.org/>`_ and `flake8 
<https://pypi.python.org/pypi/flake8>`_ are used for linting, `pytest 
<http://doc.pytest.org/en/latest/>`_ for testing, and `Sphinx 
<http://www.sphinx-doc.org/en/1.4.8/>`_ for generating the HTML documentation.

When developing, be sure to regularly run the test suite to see if anything's
broken. The suite is run automatically against Python 2 and 3 when you open a
pull request, and also when you push subsequent commits. It can be run locally
with:

.. code-block:: shell

    $ make test
    $ make code-analysis

To build and view the documentation, run ``make doc-show``.

For a list of everything you can do, run ``make help``. If you run into any
trouble, please open an issue.
