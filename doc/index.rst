probfit
=======

*probfit* is a set of functions that helps you construct a complex fit. It's
intended to be used with `iminuit <http://iminuit.github.io/iminuit/>`_. The
tool includes Binned/Unbinned Likelihood estimator, :math:`\chi^2` regression,
Binned :math:`\chi^2` estimator and Simultaneous fit estimator. Normalization and
Convolution with cache are also included. Various builtin function that's
normally used in B physics is also provided.

In a nutshell::

    import numpy as np
    from iminuit import Minuit
    from probfit import UnbinnedLH, gaussian
    data = np.random.randn(10000)
    unbinned_likelihood = UnbinnedLH(gaussian, data)
    minuit = Minuit(unbinned_likelihood, mean=0.1, sigma=1.1)
    minuit.migrad()
    unbinned_likelihood.draw(minuit)

.. toctree::
    :maxdepth: 4
    :hidden:

    api.rst
    recipe.rst

Download & Install
------------------

From pip::

    pip install probfit

or get the latest development from github::

    git clone git://github.com/iminuit/probfit.git

Tutorial
--------

The tutorial consists of an IPython notebook in the tutorial directory.
You can `view it online <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/probfit/master/tutorial/tutorial.ipynb>`_ too.


Commonly used API
-----------------

.. currentmodule:: probfit

Refer to :ref:`fullapi` for complete reference.

Cost Functions.
"""""""""""""""

Refer to :ref:`costfunc`.

.. currentmodule:: probfit.costfunc

.. autosummary::
    UnbinnedLH
    BinnedLH
    Chi2Regression
    BinnedChi2
    SimultaneousFit

Functors
""""""""

Refer to :ref:`functor`

.. currentmodule:: probfit.functor

.. autosummary::
    Normalized
    Extended
    Convolve
    AddPdf
    AddPdfNorm
    ~probfit.funcutil.rename

And corresponding decorator

.. currentmodule:: probfit.decorator

.. autosummary::
    normalized
    extended

Builtin Functions
"""""""""""""""""

Refer to :ref:`builtin`. This list can grow: implement your favorite function
and send us pull request.

.. currentmodule:: probfit.pdf

.. autosummary::
    gaussian
    crystalball
    cruijff
    cauchy
    rtv_breitwigner
    doublegaussian
    argus
    linear
    poly2
    poly3
    novosibirsk
    HistogramPdf
    Polynomial


Useful utility
""""""""""""""

You may find these functions useful in interactive environment.

.. autosummary::
    ~probfit.nputil.vector_apply
    ~probfit.plotting.draw_pdf
    ~probfit.plotting.draw_compare_hist

Cookbook
--------

Cookbook recipies are at :ref:`cookbook`.
