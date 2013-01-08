__all__ = [
    'gen_toy',
    'gen_toyn',
]

import numpy as np
from ._libstat import compute_cdf, invert_cdf, _vector_apply
from util import describe
import numpy.random as npr
from matplotlib import pyplot as plt
from probfit_warnings import SmallIntegralWarning

def gen_toyn(f, nsample, ntoy, bound, accuracy=10000, quiet=True, **kwd):
    """
    just alias of gentoy for nample and then reshape to ntoy,nsample)
    :param f:
    :param nsample:
    :param bound:
    :param accuracy:
    :param quiet:
    :param kwd:
    :return:
    """
    return gen_toy(f, nsample*ntoy, bound, accuracy, quiet, **kwd).reshape((ntoy,nsample))

def gen_toy(f, nsample, bound, accuracy=10000, quiet=True, **kwd):
    """
    generate ntoy
    :param f:
    :param nsample:
    :param ntoy:
    :param bound:
    :param accuracy:
    :param quiet:
    :param kwd: the rest of keyword argument will be passed to f
    :return: numpy.ndarray
    """
    #based on inverting cdf this is fast but you will need to give it a reasonable range
    #unlike roofit which is based on accept reject

    vnames = describe(f)
    if not quiet:
        print vnames
    my_arg = [kwd[v] for v in vnames[1:]]
    #random number
    #if accuracy is None: accuracy=10*numtoys
    r = npr.random_sample(nsample)
    x = np.linspace(bound[0], bound[1], accuracy)
    pdf = _vector_apply(f, x, tuple(my_arg))
    cdf = compute_cdf(pdf, x)
    if cdf[-1] < 0.01:
        warn(SmallIntegralWarning('Integral for given funcition is'
            ' really low. Did you give it a reasonable range?'))
    cdfnorm = cdf[-1]
    cdf /= cdfnorm

    #now convert that to toy
    ret = invert_cdf(r, cdf, x)

    if not quiet:
        #move this to plotting
        plt.figure()
        plt.title('comparison')
        numbin = 100
        h,e = np.histogram(ret,bins=numbin)
        mp = (e[1:]+e[:-1])/2.
        err = np.sqrt(h)
        plt.errorbar(mp,h,err,fmt='.b')
        bw = e[1] - e[0]
        y = pdf * len(ret) / cdfnorm * bw
        ylow = y + np.sqrt(y)
        yhigh = y - np.sqrt(y)
        plt.plot(x, y, label='pdf', color='r')
        plt.fill_between(x, yhigh, ylow, color='g', alpha=0.2)
        plt.grid(True)
        plt.xlim(bound)
        plt.ylim(ymin=0)
    return ret

