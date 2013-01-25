.. -*- mode: rst -*-

.. image:: https://travis-ci.org/iminuit/probfit.png?branch=master
   :target: https://travis-ci.org/iminuit/probfit


probfit
--------

*probfit* is a set of functions that helps you construct a complex fit. It's
intended to be used with `iminuit <http://iminuit.github.com/iminuit/>`_. The
tool includes Binned/Unbinned Likelihood estimator, :math:`\chi^2` regression,
Binned :math:`\chi^2` estimator and Simultaneous fit estimator.
Various functors for manipulating PDF such as Normalization and
Convolution(with caching) and various builtin functions
normally used in B physics is also provided.

::

    from probfit import UnbinnedLH, gaussian
    from iminuit import Minuit
    data = np.randn(10000)
    ulh = UnbinnedLH(data)
    m = Minuit(ulh, mean=0.1, sigma=1.1)
    m.migrad()
    ulh.draw(m)


Requirement
-----------

- iminuit http://iminuit.github.com/iminuit/
- numpy http://www.numpy.org/
- matplotlib http://matplotlib.org/

Tutorial
--------

open tutorial.ipynb in ipython notebook. You can `view it online <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/probfit/master/tutorial/tutorial.ipynb>`_ too.


Documentation
-------------

See `here <http://iminuit.github.com/probfit/>`_
