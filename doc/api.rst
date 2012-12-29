.. _fullapi:

Full API Documentation
======================

.. _costfunc:

Cost Function
-------------

Various estimators.

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

.. _functor:

Functor
-------

Manipulate and combined your pdf in various ways.

.. currentmodule:: probfit.functor

Extended
^^^^^^^^
.. autoclass:: Extended

Normalized
^^^^^^^^^^
.. autoclass:: Normalized

Convolve
^^^^^^^^
.. autoclass:: Convolve

AddPdf
^^^^^^
.. autoclass:: AddPdf

Add2PdfNorm
^^^^^^^^^^^
.. autoclass:: Add2PdfNorm

rename
^^^^^^

.. autofunction:: probfit.funcutil.rename

Decorator
^^^^^^^^^^
.. currentmodule:: probfit.decorator

.. autoclass:: normalized

.. autoclass:: extended

.. _builtin:

Builtin PDF
-----------

Builtin PDF written in cython.

.. currentmodule:: probfit.pdf

gaussian
^^^^^^^^
.. autofunction:: gaussian

cauchy
^^^^^^
.. autofunction:: cauchy

Breit-Wigner
^^^^^^^^^^^^
.. autofunction:: breitwigner

crystalball
^^^^^^^^^^^
.. autofunction:: crystalball

cruijff
^^^^^^^
.. autofunction:: cruijff

doublegaussian
^^^^^^^^^^^^^^
.. autofunction:: doublegaussian

novosibirsk
^^^^^^^^^^^
.. autofunction:: novosibirsk

argus
^^^^^
.. autofunction:: argus

linear
^^^^^^
.. autofunction:: linear

poly2
^^^^^
.. autofunction:: poly2

poly3
^^^^^
.. autofunction:: poly3

Polynomial
^^^^^^^^^^
.. autoclass:: Polynomial

Useful Utility Function
-----------------------
vector_apply
^^^^^^^^^^^^
.. autofunction:: probfit.nputil.vector_apply

draw_compare_hist
^^^^^^^^^^^^^^^^^
.. autofunction:: probfit.plotting.draw_compare_hist
