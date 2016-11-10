.. -*- mode: rst -*-

.. image:: https://travis-ci.org/iminuit/probfit.png?branch=master
   :target: https://travis-ci.org/iminuit/probfit


probfit
-------

*probfit* is a set of functions that helps you construct a complex fit. It's
intended to be used with `iminuit <http://iminuit.readthedocs.org/>`_. The
tool includes Binned/Unbinned Likelihood estimator, :math:`\chi^2` regression,
Binned :math:`\chi^2` estimator and Simultaneous fit estimator.
Various functors for manipulating PDF such as Normalization and
Convolution(with caching) and various builtin functions
normally used in B physics is also provided.

.. code-block:: python

    import numpy as np
    from iminuit import Minuit
    from probfit import UnbinnedLH, gaussian
    data = np.random.randn(10000)
    unbinned_likelihood = UnbinnedLH(gaussian, data)
    minuit = Minuit(unbinned_likelihood, mean=0.1, sigma=1.1)
    minuit.migrad()
    unbinned_likelihood.draw(minuit)


* `MIT <http://opensource.org/licenses/MIT>`_ license (open source)
* `Documentation <http://probfit.readthedocs.org/>`_
* The tutorial is an IPython notebook that you can view online
  `here <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/probfit/master/tutorial/tutorial.ipynb>`_.
  To run it locally: `cd tutorial; ipython notebook --pylab=inline tutorial.ipynb`.
* Dependencies:
   - `iminuit <http://iminuit.readthedocs.org/>`_
   - `numpy <http://www.numpy.org/>`_
   - `matplotlib <http://matplotlib.org/>`_ (optional, for plotting)

Development
-----------

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
broken. The suite is run automatically when you open a pull request, and when
you push subsequent commits. It can be run locally with:

.. code-block:: shell

    $ make test
    $ make code-analysis

To build and view the documentation, run ``make doc-show``.

For a list of everything you can do, run ``make help``. If you run into any
trouble, please open an issue.
