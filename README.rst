.. -*- mode: rst -*-

.. image:: https://travis-ci.org/scikit-hep/probfit.png?branch=master
   :target: https://travis-ci.org/scikit-hep/probfit


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
  `here <http://nbviewer.ipython.org/urls/raw.github.com/scikit-hep/probfit/master/tutorial/tutorial.ipynb>`_.
  To run it locally: `cd tutorial; ipython notebook --pylab=inline tutorial.ipynb`.
* Dependencies:
   - `iminuit <http://iminuit.readthedocs.org/>`_
   - `numpy <http://www.numpy.org/>`_
   - `matplotlib <http://matplotlib.org/>`_ (optional, for plotting)
* Developing probfit: see the `development page <http://probfit.readthedocs.io/en/latest/development.html>`_
