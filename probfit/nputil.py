import numpy as np
from _libstat import _vector_apply


def mid(x):
    return (x[:-1]+x[1:])/2


def minmax(x):
    return min(x), max(x)


#for converting numpy array to double
def float2double(a):
    if a is None or a.dtype == np.float64:
        return a
    else:
        return a.astype(np.float64)


def vector_apply(f, x, *arg):
    """
    apply **f** to array **x** with given arguments fast. This is a fast
    version of::

        np.array([f(xi,*arg) for xi in x ])

    useful when you try to plot something
    """
    return _vector_apply(f, x, arg)
