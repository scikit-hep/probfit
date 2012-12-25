.. dist_fit documentation master file, created by
   sphinx-quickstart on Sat Nov 10 11:16:37 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dist_fit's documentation!
====================================

Tutorial
--------

lorem ipsum

API
---

.. currentmodule:: probfit

Summary
^^^^^^^


Cost Functions.
"""""""""""""""

.. currentmodule:: probfit.cdist_fit

.. autosummary::
    UnbinnedLH
    BinnedLH
    Chi2Regression
    BinnedChi2

Builtin Functions
"""""""""""""""""

.. currentmodule:: probfit.cdist_func

.. autosummary::
    gaussian
    crystalball
    cruijff
    doublegaussian
    argus
    linear
    poly2
    poly3
    novosibirsk

Functors
""""""""

.. currentmodule:: probfit.cdist_func

.. autosummary::
    Normalized
    Extend
    Convolve

Unbinned Likelihood
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: probfit.cdist_fit

.. autoclass:: UnbinnedLH

    .. automethod:: __call__
    .. automethod:: draw
    .. automethod:: show

Binned Likelihood
^^^^^^^^^^^^^^^^^

.. autoclass:: BinnedLH

    .. automethod:: __call__
    .. automethod:: draw
    .. automethod:: show

:math:`\chi^2` Regression
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Chi2Regression

    .. automethod:: __call__
    .. automethod:: draw
    .. automethod:: show

Binned :math:`\chi^2`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BinnedChi2

    .. automethod:: __call__
    .. automethod:: draw
    .. automethod:: show

Functor
-------

Builtin PDF
-----------

.. currentmodule:: probfit.cdist_func

.. autofunction:: gaussian
.. autofunction:: crystalball
.. autofunction:: cruijff
.. autofunction:: doublegaussian
.. autofunction:: argus
.. autofunction:: linear
.. autofunction:: poly2
.. autofunction:: poly3
.. autofunction:: novosibirsk
