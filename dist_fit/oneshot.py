from .cdist_fit import UnbinnedML, BinnedChi2, compute_cdf, invert_cdf, BinnedLH
from RTMinuit import Minuit
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as npr
import itertools as itt
import collections
from warnings import warn
from dist_fit.util import parse_arg
from dist_fit.common import minmax
from util import vertical_highlight, better_arg_spec

def randfr(r):
    """
    generate a uniform random number with in range r
    :param r: tuple range
    :return: float
    """
    b = r[1]
    a = r[0]
    return np.random.ranf() * (b - a) + a


def fit_uml(f, data, quiet=False, printlevel=0, *arg, **kwd):
    """
    perform unbinned likelihood fit
    :param f: pdf
    :param data: data
    :param quiet: if not quitet draw latest fit on fail fit
    :param printlevel: minuit printlevel
    :return:
    """
    uml = UnbinnedML(f, data)
    m = Minuit(uml, printlevel=printlevel, **kwd)
    m.set_strategy(2)
    m.set_up(0.5)
    m.migrad()
    if not m.migrad_ok() or not m.matrix_accurate():
        if not quiet:
            plt.figure()
            uml.show()
            print m.values
    return (uml, m)


def fit_binx2(f, data, bins=30, range=None, printlevel=0, quiet=False, *arg, **kwd):
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
    uml = BinnedChi2(f, data, bins=bins, range=range)
    m = Minuit(uml, printlevel=printlevel, **kwd)
    m.set_strategy(2)
    m.set_up(1)
    m.migrad()
    if not m.migrad_ok() or not m.matrix_accurate():
        if not quiet:
            plt.figure()
            uml.show()
            print m.values

    return (uml, m)


def fit_binlh(f, data, bins=30,
              range=None, quiet=False, weights=None, use_w2=False,
              printlevel=0, pedantic=True, extended=False,
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
    uml = BinnedLH(f, data, bins=bins, range=range, weights=weights, use_w2=use_w2, extended=extended)
    m = Minuit(uml, printlevel=printlevel, pedantic=pedantic, **kwd)
    m.set_strategy(2)
    m.set_up(0.5)
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
    fom = UnbinnedML(f, data)
    narg = f.func_code.co_argcount
    vnames = f.func_code.co_varnames[1:narg]
    my_arg = [tuplize(kwd[name]) for name in vnames]
    h, e, _ = plt.hist(data, bins=bins, normed=True, histtype='step')
    vf = np.vectorize(f)
    vx = np.linspace(e[0], e[-1], fbins)
    first = True
    minfom = 0
    minarg = None
    for thisarg in itt.product(*my_arg):
        vy = vf(vx, *thisarg)
        plt.plot(vx, vy, '-', label=pprint_arg(vnames, thisarg))
        thisfom = fom(*thisarg)
        if first or thisfom < minfom:
            minfom = thisfom
            minarg = thisarg
            first = False
    leg = plt.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ret = {k: v for k, v in zip(vnames, minarg)}
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
    vf = np.vectorize(f)
    vx = np.linspace(e[0], e[-1], fbins)
    first = True
    minfom = 0
    minarg = None
    for thisarg in itt.product(*my_arg):
        vy = vf(vx, *thisarg) * bw
        plt.plot(vx, vy, '-', label=pprint_arg(vnames, thisarg))
        thisfom = fom(*thisarg)
        if first or thisfom < minfom:
            minfom = thisfom
            minarg = thisarg
            first = False
    leg = plt.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ret = {k: v for k, v in zip(vnames, minarg)}
    return ret


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

    vnames = better_arg_spec(f)
    if not quiet:
        print vnames
    my_arg = [kwd[v] for v in vnames[1:]]
    #random number
    #if accuracy is None: accuracy=10*numtoys
    r = npr.random_sample(nsample)
    x = np.linspace(bound[0], bound[1], accuracy)
    vf = np.vectorize(f)
    pdf = vf(x, *my_arg)
    cdf = compute_cdf(pdf, x)
    if cdf[-1] < 0.01:
        warn('Integral for given funcition is really low. Did you give it a reasonable range?')
    cdfnorm = cdf[-1]
    cdf /= cdfnorm

    #now convert that to toy
    ret = invert_cdf(r, cdf, x)

    if not quiet:
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


def guess_initial(alg, f, data, ntry=100, guessrange=(-100, 100), draw=False, *arg, **kwd):
    """
    This is very bad at the moment don't use it
    """
    fom = alg(f, data, *arg, **kwd)
    first = True
    minfom = 0
    minparam = ()
    narg = fom.func_code.co_argcount
    vnames = fom.func_code.co_varnames[:narg]
    ranges = {}
    for vname in vnames:
        if 'limit_' + vname in kwd:
            ranges[vname] = kwd['limit_' + vname]
        else:
            ranges[vname] = guessrange

    for i in xrange(ntry):
        arg = []
        for vname in vnames:
            arg.append(randfr(ranges[vname]))
        try:
            thisfom = fom(*arg)
            if first or thisfom < minfom:
                first = False
                minfom = thisfom
                minparam = arg
        except ZeroDivisionError:
            pass

    print minparam, minfom
    ret = {}
    for vname, bestguess in zip(vnames, minparam):
        ret[vname] = bestguess
    if draw:
        fom(*minparam)
        fom.draw()
    return ret


def draw_contour2d(fit, m, var1, var2, bins=12, bound1=None, bound2=None, lh=True):
    x1s, x2s, y = val_contour2d(fit, m, var1, var2, bins=bins, bound1=bound1, bound2=bound2)
    y -= np.min(y)
    v = np.array([0.5, 1, 1.5])
    if not lh: v *= 2
    CS = plt.contour(x1s, x2s, y, v, colors=['b', 'k', 'r'])
    plt.clabel(CS)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.axhline(m.values[var2], color='k', ls='--')
    plt.axvline(m.values[var1], color='k', ls='--')
    plt.grid(True)
    return x1s, x2s, y, CS

def draw_compare(f, arg, edges, data, errors=None, normed=False, parts=False):
    #arg is either map or tuple
    if isinstance(arg, dict): arg = parse_arg(f,arg,1)
    x = (edges[:-1]+edges[1:])/2.0
    bw = np.diff(edges)
    vf = np.vectorize(f)
    yf = vf(x,*arg)
    total = np.sum(data)
    if normed:
        plt.errorbar(x,data/bw/total,errors/bw/total,fmt='.b')
        plt.plot(x,yf,'r',lw=2)
    else:
        plt.errorbar(x,data,errors,fmt='.b')
        plt.plot(x,yf*bw,'r',lw=2)

    #now draw the parts
    if parts:
        if not hasattr(f,'nparts') or not hasattr(f,'eval_parts'):
            warn('parts is set to True but function does not have nparts or eval_parts method')
        else:
            scale = bw if not normed else 1.
            nparts = f.nparts
            parts_val = list()
            for tx in x:
                val = f.eval_parts(tx,*arg)
                parts_val.append(val)
            py = zip(*parts_val)
            for y in py:
                tmpy = np.array(y)
                plt.plot(x,tmpy*scale,lw=2,alpha=0.5)
    plt.grid(True)
    return x,yf,data

def draw_pdf(f,arg,bound,bins,scale=1.0,normed=True,**kwds):
    edges = np.linspace(bound[0],bound[1],bins)
    return draw_pdf_with_edges(f,arg,edges,scale=scale,normed=normed,**kwds)

def draw_pdf_with_edges(f,arg,edges,scale=1.0,normed=True,**kwds):
    if isinstance(arg, dict): arg = parse_arg(f,arg,1)
    x = (edges[:-1]+edges[1:])/2.0
    bw = np.diff(edges)
    vf = np.vectorize(f)
    yf = vf(x,*arg)
    scale*= bw if not normed else 1.
    yf*=scale
    plt.plot(x,yf,lw=2,**kwds)
    return x,yf

#draw comprison between function given args and data
def draw_compare_hist(f, arg, data, bins=100, bound=None,weights=None, normed=False, use_w2=False, parts=False):
    if bound is None: bound = minmax(data)
    h,e = np.histogram(data,bins=bins,range=bound,weights=weights)
    err = None
    if weights is not None and use_w2:
       err,_ = np.histogram(data,bins=bins,range=bound,weights=weights*weights)
       err = np.sqrt(err)
    else:
        err = np.sqrt(h)
    return draw_compare(f,arg,e,h,err,normed,parts)

def safe_getattr(o,attr,default=None):
    if hasattr(o,attr):
        return getattr(o,attr)
    else:
        return default

def draw_compare_fit_statistics(lh, m):
    pass


def val_contour2d(fit, m, var1, var2, bins=12, bound1=None, bound2=None):
    assert(var1 != var2)
    if bound1 is None: bound1 = (m.values[var1] - 2 * m.errors[var1], m.values[var1] + 2 * m.errors[var1])
    if bound2 is None: bound2 = (m.values[var2] - 2 * m.errors[var2], m.values[var2] + 2 * m.errors[var2])

    x1s = np.linspace(bound1[0], bound1[1], bins)
    x2s = np.linspace(bound2[0], bound2[1], bins)
    ys = np.zeros((bins, bins))

    x1pos = m.var2pos[var1]
    x2pos = m.var2pos[var2]

    arg = list(m.args)

    for ix1 in xrange(len(x1s)):
        arg[x1pos] = x1s[ix1]
        for ix2 in xrange(len(x2s)):
            arg[x2pos] = x2s[ix2]
            ys[ix1, ix2] = fit(*arg)
    return x1s, x2s, ys


def draw_contour(fit, m, var, bound=None, step=None, lh=True):
    x, y = val_contour(fit, m, var, bound=bound, step=step)
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel(var)
    if lh: plt.ylabel('-Log likelihood')

    minpos = np.argmin(y)
    ymin = y[minpos]
    tmpy = y - ymin
    #now scan for minpos to the right until greater than one
    up = {True: 0.5, False: 1.}[lh]
    righty = np.power(tmpy[minpos:] - up, 2)
    #print righty
    right_min = np.argmin(righty)
    rightpos = right_min + minpos
    lefty = np.power((tmpy[:minpos] - up), 2)
    left_min = np.argmin(lefty)
    leftpos = left_min
    print leftpos, rightpos
    le = x[minpos] - x[leftpos]
    re = x[rightpos] - x[minpos]
    print (le, re)
    vertical_highlight(x[leftpos], x[rightpos])
    plt.figtext(0.5, 0.5, '%s = %7.3e ( -%7.3e , +%7.3e)' % (var, x[minpos], le, re), ha='center')


def val_contour(fit, m, var, bound=None, step=None):
    arg = list(m.args)
    pos = m.var2pos[var]
    val = m.values[var]
    error = m.errors[var]
    if bound is None: bound = (val - 2. * error, val + 2. * error)
    if step is None: step = error / 100.

    xs = np.arange(bound[0], bound[1], step)
    ys = np.zeros(len(xs))
    for i, x in enumerate(xs):
        arg[pos] = x
        ys[i] = fit(*arg)

    return xs, ys
