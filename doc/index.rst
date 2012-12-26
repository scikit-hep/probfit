Welcome to probfit's documentation!
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

.. currentmodule:: probfit.costfunc

.. autosummary::
    UnbinnedLH
    BinnedLH
    Chi2Regression
    BinnedChi2

Builtin Functions
"""""""""""""""""

.. currentmodule:: probfit.pdf

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

.. currentmodule:: probfit.functor

.. autosummary::
    Normalized
    Extended
    Convolve
    AddPdf
    Add2PdfNorm

Unbinned Likelihood
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: probfit.costfunc

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

.. currentmodule:: probfit.functor


.. autoclass:: Extended
.. autoclass:: Normalized
.. autoclass:: Convolve
.. autoclass:: AddPdf
.. autoclass:: Add2PdfNorm

Builtin PDF
-----------

.. currentmodule:: probfit.pdf

.. autofunction:: gaussian
.. autofunction:: crystalball
.. autofunction:: cruijff
.. autofunction:: doublegaussian
.. autofunction:: novosibirsk
.. autofunction:: argus
.. autofunction:: linear
.. autofunction:: poly2
.. autofunction:: poly3
.. autoclass:: Polynomial

