probfit
=======

*probfit* is a set of functions that helps you construct a complex fit. It's
intended to be used with `iminuit <http://iminuit.github.com/iminuit/>`_. The
tool includes Binned/Unbinned Likelihood estimator, :math:`\chi^2` regression,
Binned :math:`\chi^2` estimator and Simultaneous fit estimator. Normalization and
Convolution with cache are also included. Various builtin function that's
normally used in B physics is also provided.

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

Tutorial is in the tutorial directory. You can `view it online <http://nbviewer.ipython.org/urls/raw.github.com/piti118/probfit/master/tutorial/tutorial.ipynb>`_ too.


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

