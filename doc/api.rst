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

**Example**

    .. plot:: pyplots/costfunc/ulh.py
        :class: lightbox

Binned Likelihood
^^^^^^^^^^^^^^^^^

.. autoclass:: BinnedLH

    .. automethod:: __call__
    .. automethod:: draw
    .. automethod:: show

**Example**

    .. plot:: pyplots/costfunc/blh.py
        :class: lightbox

:math:`\chi^2` Regression
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Chi2Regression

    .. automethod:: __call__
    .. automethod:: draw
    .. automethod:: show

**Example**

    .. plot:: pyplots/costfunc/x2r.py
        :class: lightbox

Binned :math:`\chi^2`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BinnedChi2

    .. automethod:: __call__
    .. automethod:: draw
    .. automethod:: show

**Example**


    .. plot:: pyplots/costfunc/bx2.py
        :class: lightbox

Simultaneous Fit
^^^^^^^^^^^^^^^^

.. autoclass:: SimultaneousFit

    .. automethod:: __call__
    .. automethod:: args_and_error_for
    .. automethod:: draw
    .. automethod:: show

**Example**

    .. plot:: pyplots/costfunc/simul.py

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

BlindFunc
^^^^^^^^^
.. autoclass:: BlindFunc

AddPdf
^^^^^^
.. autoclass:: AddPdf

**Example**

    .. plot:: pyplots/functor/addpdf.py
        :class: lightbox

AddPdfNorm
^^^^^^^^^^^
.. autoclass:: AddPdfNorm

**Example**

    .. plot:: pyplots/functor/addpdfnorm.py
        :class: lightbox

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

.. plot:: pyplots/pdf/gaussian.py
    :class: lightbox

cauchy
^^^^^^
.. autofunction:: cauchy

Breit-Wigner
^^^^^^^^^^^^
.. autofunction:: rtv_breitwigner

crystalball
^^^^^^^^^^^
.. autofunction:: crystalball

.. plot:: pyplots/pdf/crystalball.py
    :class: lightbox

cruijff
^^^^^^^
.. autofunction:: cruijff

.. plot:: pyplots/pdf/cruijff.py
    :class: lightbox

doublegaussian
^^^^^^^^^^^^^^
.. autofunction:: doublegaussian

.. plot:: pyplots/pdf/doublegaussian.py
    :class: lightbox

novosibirsk
^^^^^^^^^^^
.. autofunction:: novosibirsk

.. plot:: pyplots/pdf/novosibirsk.py
    :class: lightbox

argus
^^^^^
.. autofunction:: argus

.. plot:: pyplots/pdf/argus.py
    :class: lightbox

linear
^^^^^^
.. autofunction:: linear

.. plot:: pyplots/pdf/linear.py
    :class: lightbox

poly2
^^^^^
.. autofunction:: poly2
.. plot:: pyplots/pdf/poly2.py
    :class: lightbox

poly3
^^^^^
.. autofunction:: poly3

.. plot:: pyplots/pdf/poly3.py
    :class: lightbox

Polynomial
^^^^^^^^^^
.. autoclass:: Polynomial

.. plot:: pyplots/pdf/polynomial.py
    :class: lightbox

HistogramPdf
^^^^^^^^^^^^
.. autoclass:: HistogramPdf

.. plot:: pyplots/pdf/histogrampdf.py
    :class: lightbox

Useful Utility Function
-----------------------
vector_apply
^^^^^^^^^^^^
.. autofunction:: probfit.nputil.vector_apply

draw_pdf
^^^^^^^^
.. autofunction:: probfit.plotting.draw_pdf

draw_compare_hist
^^^^^^^^^^^^^^^^^
.. autofunction:: probfit.plotting.draw_compare_hist
