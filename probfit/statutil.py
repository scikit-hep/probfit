import numpy as np
from _libstat import _vector_apply


def fwhm_f(f, range, arg=None, bins=1000):

    if arg is None: arg=tuple()

    x = np.linspace(range[0], range[1], bins)
    y = _vector_apply(f, x, arg)
    imax = np.argmax(y)
    ymax = y[imax]
    rs = y[imax:]-ymax/2.0
    ls = y[:imax]-ymax/2.0
    xl = 0
    xr = 0

    il = first_neg(ls,'l')
    #print il,x[il],ls[il],x[il+1],ls[il+1]
    xl = xintercept(x[il], ls[il], x[il+1], ls[il+1])

    ir = first_neg(rs, 'r')
    #print ir,x[imax+ir],rs[ir],x[ir+1],rs[ir+1]
    xr = xintercept(x[imax+ir], rs[ir], x[imax+ir-1], rs[ir-1])

    return (xl, xr)

def xintercept_tuple(t0, t1):
    return xintercept(t0[0],t0[1],t1[0],t1[1])

def xintercept(x0, y0, x1, y1):
    m = (y1-y0)/(x1-x0)
    return -y0/m+x0

def first_neg(y, direction='r'):
    if direction == 'l':
        xlist = xrange(len(y)-1, -1, -1)
    else:
        xlist = xrange(len(y))
    ret = 0
    found = False
    for i in xlist:
        if y[i] < 0:
            ret = i
            found = True
            break
    if not found:
        raise ValueError('They are all positive what are you tying to find?')
    return ret
