from .costfunc import UnbinnedLH, BinnedChi2, BinnedLH
from ._libstat import _vector_apply
from iminuit import Minuit
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as npr
import itertools as itt
import collections
from warnings import warn
from .util import parse_arg, describe
from .nputil import minmax


def fit_uml(f, data, quiet=False, print_level=0, *arg, **kwd):
    """
    perform unbinned likelihood fit
    :param f: pdf
    :param data: data
    :param quiet: if not quite draw latest fit on fail fit
    :param printlevel: minuit printlevel
    :return:
    """
    uml = UnbinnedLH(f, data)
    m = Minuit(uml, print_level=print_level, **kwd)
    m.set_strategy(2)
    m.migrad()
    if not m.migrad_ok() or not m.matrix_accurate():
        if not quiet:
            plt.figure()
            uml.show()
            print m.values
    return (uml, m)


def fit_binx2(f, data, bins=30, bound=None, print_level=0, quiet=False, *arg, **kwd):
    """
    perform chi^2 fit
    :param f:
    :param data:
    :param bins:
    :param range:
    :param printlevel:
    :param quiet:
    :param arg:
    :param kwd:
    :return:
    """
    uml = BinnedChi2(f, data, bins=bins, bound=bound)
    m = Minuit(uml, print_level=print_level, **kwd)
    m.set_strategy(2)
    m.migrad()
    if not m.migrad_ok() or not m.matrix_accurate():
        if not quiet:
            plt.figure()
            uml.show()
            print m.values

    return (uml, m)


def fit_binlh(f, data, bins=30,
              bound=None, quiet=False, weights=None, use_w2=False,
              print_level=0, pedantic=True, extended=False,
              *arg, **kwd):
    """
    perform bin likelihood fit
    :param f:
    :param data:
    :param bins:
    :param range:
    :param quiet:
    :param weights:
    :param use_w2:
    :param printlevel:
    :param pedantic:
    :param extended:
    :param arg:
    :param kwd:
    :return:
    """
    uml = BinnedLH(f, data, bins=bins, bound=bound,
                    weights=weights, use_w2=use_w2, extended=extended)
    m = Minuit(uml, print_level=print_level, pedantic=pedantic, **kwd)
    m.set_strategy(2)
    m.migrad()
    if not m.migrad_ok() or not m.matrix_accurate():
        if not quiet:
            plt.figure()
            uml.show()
            print m.values
        #m.minos()
    return (uml, m)


def tuplize(x):
    """
    :param x:
    :return:
    """
    if  isinstance(x, collections.Iterable):
        return x
    else:
        return tuple([x])


def pprint_arg(vnames, value):
    """
    pretty print argument
    :param vnames:
    :param value:
    :return:
    """
    ret = ''
    for name, v in zip(vnames, value):
        ret += '%s=%s;' % (name, str(v))
    return ret;


def try_uml(f, data, bins=40, fbins=1000, *arg, **kwd):
    fom = UnbinnedLH(f, data)
    narg = f.func_code.co_argcount
    vnames = f.func_code.co_varnames[1:narg]
    my_arg = [tuplize(kwd[name]) for name in vnames]
    h, e, _ = plt.hist(data, bins=bins, normed=True, histtype='step')
    vx = np.linspace(e[0], e[-1], fbins)
    first = True
    minfom = 0
    minarg = None
    for thisarg in itt.product(*my_arg):
        vy = _vector_apply(f, vx, thisarg)
        plt.plot(vx, vy, '-', label=pprint_arg(vnames, thisarg))
        thisfom = fom(*thisarg)
        if first or thisfom < minfom:
            minfom = thisfom
            minarg = thisarg
            first = False
    leg = plt.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ret = dict((k,v) for k, v in zip(vnames, minarg))
    return ret

def try_binlh(f, data, weights=None, bins=40, fbins=1000, show='both', extended=False,
        bound=None, *arg, **kwd):
    if bound is None: bound = minmax(data)
    fom = BinnedLH(f, data,extended=extended,bound=bound)
    narg = f.func_code.co_argcount
    vnames = f.func_code.co_varnames[1:narg]
    my_arg = [tuplize(kwd[name]) for name in vnames]
    h, e = None, None
    if show == 'both':
        h, e, _ = plt.hist(data, bins=bins, range=bound, histtype='step', weights=weights, normed=not extended)
    else:
        h, e = np.histogram(data, bins=bins, range=bound, weights=weights, normed=not extended)
    bw = e[1] - e[0]

    vx = np.linspace(e[0], e[-1], fbins)
    first = True
    minfom = 0
    minarg = None
    for thisarg in itt.product(*my_arg):
        vy = _vector_apply(f, vx, thisarg)
        if extended: vy*=bw
        plt.plot(vx, vy, '-', label=pprint_arg(vnames, thisarg))
        thisfom = fom(*thisarg)
        if first or thisfom < minfom:
            minfom = thisfom
            minarg = thisarg
            first = False
    leg = plt.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ret = dict((k,v) for k, v in zip(vnames, minarg))
    return ret


def try_chi2(f, data, weights=None, bins=40, fbins=1000, show='both', *arg, **kwd):
    fom = BinnedChi2(f, data)
    narg = f.func_code.co_argcount
    vnames = f.func_code.co_varnames[1:narg]
    my_arg = [tuplize(kwd[name]) for name in vnames]
    h, e = None, None
    if show == 'both':
        h, e, _ = plt.hist(data, bins=bins, histtype='step', weights=weights)
    else:
        h, e = np.histogram(data, bins=bins, weights=weights)
    bw = e[1] - e[0]
    vx = np.linspace(e[0], e[-1], fbins)
    first = True
    minfom = 0
    minarg = None
    for thisarg in itt.product(*my_arg):
        vy = _vector_apply(f, vx, thisarg) * bw
        plt.plot(vx, vy, '-', label=pprint_arg(vnames, thisarg))
        thisfom = fom(*thisarg)
        if first or thisfom < minfom:
            minfom = thisfom
            minarg = thisarg
            first = False
    leg = plt.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ret = dict((k, v) for k, v in zip(vnames, minarg))
    return ret


# def randfr(r):
#     """
#     generate a uniform random number with in range r
#     :param r: tuple range
#     :return: float
#     """
#     b = r[1]
#     a = r[0]
#     return np.random.ranf() * (b - a) + a

# def guess_initial(alg, f, data, ntry=100, guessrange=(-100, 100), draw=False, *arg, **kwd):
#     """
#     This is very bad at the moment don't use it
#     """
#     fom = alg(f, data, *arg, **kwd)
#     first = True
#     minfom = 0
#     minparam = ()
#     narg = fom.func_code.co_argcount
#     vnames = fom.func_code.co_varnames[:narg]
#     ranges = {}
#     for vname in vnames:
#         if 'limit_' + vname in kwd:
#             ranges[vname] = kwd['limit_' + vname]
#         else:
#             ranges[vname] = guessrange

#     for i in xrange(ntry):
#         arg = []
#         for vname in vnames:
#             arg.append(randfr(ranges[vname]))
#         try:
#             thisfom = fom(*arg)
#             if first or thisfom < minfom:
#                 first = False
#                 minfom = thisfom
#                 minparam = arg
#         except ZeroDivisionError:
#             pass

#     print minparam, minfom
#     ret = {}
#     for vname, bestguess in zip(vnames, minparam):
#         ret[vname] = bestguess
#     if draw:
#         fom(*minparam)
#         fom.draw()
#     return ret
