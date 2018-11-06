.. -*- mode: rst -*-

probfit
=======

.. image:: https://img.shields.io/pypi/v/probfit.svg
   :target: https://pypi.python.org/pypi/probfit

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1477853.svg
   :target: https://doi.org/10.5281/zenodo.1477853

.. image:: https://travis-ci.org/scikit-hep/probfit.png?branch=master
   :target: https://travis-ci.org/scikit-hep/probfit

*probfit* is a set of functions that helps you construct a complex fit. It's
intended to be used with `iminuit <http://iminuit.readthedocs.org/>`_. The
tool includes Binned/Unbinned Likelihood estimators, :math:`\chi^2` regression,
Binned :math:`\chi^2` estimator and Simultaneous fit estimator.
Various functors for manipulating PDFs such as Normalization and
Convolution (with caching) and various built-in functions
normally used in B physics are also provided.

Strict dependencies
-------------------

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`__ (2.7+, 3.4+)
- `Numpy <https://scipy.org/install.html>`__
- `iminuit <http://iminuit.readthedocs.org/>`_

Optional dependencies
---------------------

- `matplotlib <http://matplotlib.org/>`_ for the plotting functions

Getting started
---------------

.. code-block:: python

    import numpy as np
    from iminuit import Minuit
    from probfit import UnbinnedLH, gaussian
    data = np.random.randn(10000)
    unbinned_likelihood = UnbinnedLH(gaussian, data)
    minuit = Minuit(unbinned_likelihood, mean=0.1, sigma=1.1)
    minuit.migrad()
    unbinned_likelihood.draw(minuit)

Documentation and Tutorial
--------------------------

* `Documentation <http://probfit.readthedocs.org/>`_
* The tutorial is an IPython notebook that you can view online
  `here <http://nbviewer.ipython.org/urls/raw.github.com/scikit-hep/probfit/master/tutorial/tutorial.ipynb>`_.
  To run it locally: `cd tutorial; ipython notebook --pylab=inline tutorial.ipynb`.
* Developing probfit: see the `development page <http://probfit.readthedocs.io/en/latest/development.html>`_

License
-------

The package is licensed under the `MIT <http://opensource.org/licenses/MIT>`_ license (open source).
