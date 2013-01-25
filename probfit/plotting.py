 # -*- coding: utf-8 -*-
#Plotting is on python since this will make it much easier to debug and adjsut
#no need to recompile everytime i change graph color....

#needs a serious refactor
from matplotlib import pyplot as plt
import numpy as np
from .nputil import mid, minmax, vector_apply
from util import parse_arg, describe
from math import sqrt, ceil, floor
from warnings import warn


def draw_simultaneous(self, minuit=None, args=None, errors=None, **kwds):
    numf = len(self.allf)
    numraw = sqrt(numf)
    numcol = ceil(numraw)
    numrow = floor(numraw) if floor(numraw)*numcol>=numf else ceil(numraw)

    for i in range(numf):
        plt.subplot(numrow, numcol, i+1)
        part_args, part_errors = self.args_and_error_for(i, minuit, args, errors)
        self.allf[i].draw(args=part_args, errors=part_errors, **kwds)


def _get_args_and_errors(self, minuit=None, args=None, errors=None):
    """
    consitent algorithm to get argument and errors
    1) get it from minuit if minuit is avaialble
    2) if not get it from args and errors
    2.1) if args is dict parse it.
    3) if all else fail get it from self.last_arg
    """
    ret_arg = None
    ret_error = None
    if minuit is not None: # case 1
        ret_arg = minuit.args
        ret_error = minuit.errors
        return ret_arg, ret_error

    #no minuit specified use args and errors
    if args is not None:
        if isinstance(args, dict):
            ret_arg = parse_arg(self, args)
        else:
            ret_arg = args
    else: # case 3
        ret_arg = self.last_arg

    if errors is not None:
        ret_error = errors

    return ret_arg, ret_error


def _param_text(parameters, arg, error):
    txt = u''
    for i, (k, v) in enumerate(zip(parameters, arg)):
        txt += u'%s = %5.4g'%(k, v)
        if error is not None:
            txt += u'±%5.4g'%error[k]
        txt += u'\n'
    return txt


#from UML
def draw_ulh(self, minuit=None, bins=100, ax=None, bound=None,
            parmloc=(0.05, 0.95), nfbins=500, print_par=True,
            args=None, errors=None, parts=False):

    ax = plt.gca() if ax is None else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    n, e, patches = ax.hist(self.data, bins=bins, weights=self.weights,
                            histtype='step', range=bound,
                            normed=not self.extended)

    #bound = (e[0], e[-1])
    draw_arg = [('lw', 2)]
    if not parts:
        draw_arg.append(('color', 'r'))

    draw_pdf_with_edges(self.f, arg, e, density=not self.extended,
                    **dict(draw_arg))

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                draw_pdf_with_edges(p, arg, e, density=not self.extended)

    ax.grid(True)

    txt = _param_text(describe(self), arg, error)
    if print_par:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)


#from chi2 regression
def draw_x2(self, minuit=None, ax=None, parmloc=(0.05, 0.95), print_par=True,
            args=None, errors=None):

    ax = plt.gca() if ax is None else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    x=self.x
    y=self.y
    data_err = self.error

    if data_err is None:
        plt.plot(x, y, '+')
    else:
        plt.errorbar(x, y, data_err, fmt='.')

    draw_arg = [('lw', 2)]
    draw_arg.append(('color', 'r'))

    draw_pdf_with_midpoints(self.f, arg, x, **dict(draw_arg))

    ax.grid(True)

    txt = _param_text(describe(self), arg, error)

    chi2 = self(*arg)
    txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2/self.ndof, chi2, self.ndof)

    if print_par:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
            transform=ax.transAxes)


#from binned chi2
def draw_bx2(self, minuit=None, parmloc=(0.05, 0.95), nfbins=500, ax=None,
             print_par=True, args=None, errors=None, parts=False):

    ax = ax=plt.gca() if ax is None else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    m = mid(self.edges)

    ax.errorbar(m, self.h, self.err, fmt='.')

    bound = (self.edges[0], self.edges[-1])

    scale = nfbins/self.bins #scale back to bins

    draw_arg = [('lw', 2)]

    if not parts:
        draw_arg.append(('color', 'r'))

    draw_pdf(self.f, arg, bins=nfbins, bound=bound, density=False,
             scale=scale, **dict(draw_arg))

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                draw_pdf(p, arg, bound=bound, bins=nfbins, density=False,
                         scale=scale)

    ax.grid(True)

    txt = _param_text(describe(self), arg, error)

    chi2 = self(*arg)
    txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2/self.ndof, chi2, self.ndof)

    if print_par:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)


#from binnedLH
def draw_blh(self, minuit=None, parmloc=(0.05, 0.95),
                nfbins=1000, ax=None, print_par=True,
                args=None, errors=None, parts=False):
    ax = ax=plt.gca() if ax is None else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    m = mid(self.edges)

    if self.use_w2:
        err = np.sqrt(self.w2)
    else:
        err = np.sqrt(self.h)

    if self.extended:
        ax.errorbar(m, self.h, err, fmt='.')
    else:
        scale = 1./sum(self.h)/self.binwidth
        ax.errorbar(m, self.h*scale, err*scale, fmt='.')

    draw_arg = [('lw', 2)]
    if not parts:
        draw_arg.append(('color', 'r'))
    bound = (self.edges[0], self.edges[-1])
    scale = 1. if not self.extended else nfbins/self.bins #scale back to bins
    draw_pdf(self.f, arg, bins=nfbins, bound=bound, density=not self.extended,
             scale=scale, **dict(draw_arg))

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                draw_pdf(p, arg, bins=nfbins, bound=bound,
                         density=not self.extended, scale=scale)

    ax.grid(True)

    txt = _param_text(describe(self), arg, error)

    if print_par:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
            transform=ax.transAxes)


def draw_compare(f, arg, edges, data, errors=None, normed=False, parts=False):
    """
    this need to be rewritten
    """
    #arg is either map or tuple
    arg = parse_arg(f, arg, 1) if isinstance(arg, dict) else arg
    x = (edges[:-1]+edges[1:])/2.0
    bw = np.diff(edges)
    yf = vector_apply(f, x, *arg)
    total = np.sum(data)
    if normed:
        plt.errorbar(x, data/bw/total, errors/bw/total, fmt='.b')
        plt.plot(x, yf, 'r', lw=2)
    else:
        plt.errorbar(x, data, errors, fmt='.b')
        plt.plot(x, yf*bw, 'r', lw=2)

    #now draw the parts
    if parts:
        if not hasattr(f, 'eval_parts'):
            warn(RuntimeWarning('parts is set to True but function does '
                            'not have eval_parts method'))
        else:
            scale = bw if not normed else 1.
            parts_val = list()
            for tx in x:
                val = f.eval_parts(tx, *arg)
                parts_val.append(val)
            py = zip(*parts_val)
            for y in py:
                tmpy = np.array(y)
                plt.plot(x, tmpy*scale, lw=2, alpha=0.5)
    plt.grid(True)
    return x, yf, data


def draw_normed_pdf(f, arg, bound, bins=100, scale=1.0, density=True, **kwds):
        return draw_pdf(f, arg, bound, bins=100, scale=1.0, density=True,
                        normed_pdf=True, **kwds)


def draw_pdf(f, arg, bound, bins=100, scale=1.0, density=True,
             normed_pdf=False, **kwds):
    """
    draw pdf with given argument and bounds.

    **Arguments**

        * **f** your pdf. The first argument is assumed to be independent
          variable

        * **arg** argument can be tuple or list

        * **bound** tuple(xmin,xmax)

        * **bins** number of bins to plot pdf. Default 100.

        * **scale** multiply pdf by given number. Default 1.0.

        * **density** plot density instead of expected count in each bin
          (pdf*bin width). Default True.

        * **normed_pdf** Normalize pdf in given bound. Default False

        * The rest of keyword argument will be pass to pyplot.plot

    **Returns**
    
        x, y of what's being plot
    """
    edges = np.linspace(bound[0], bound[1], bins)
    return draw_pdf_with_edges(f, arg, edges, scale=scale, density=density,
                               normed_pdf=normed_pdf, **kwds)


def draw_pdf_with_edges(f, arg, edges, scale=1.0, density=True,
                        normed_pdf=False, **kwds):
    x = (edges[:-1]+edges[1:])/2.0
    bw = np.diff(edges)
    scale *= bw if not density else 1.
    return draw_pdf_with_midpoints(f, arg, x, scale=scale,
                                   normed_pdf=normed_pdf, **kwds)


def draw_pdf_with_midpoints(f, arg, x, scale=1.0, normed_pdf=False, **kwds):
    arg = parse_arg(f, arg, 1) if isinstance(arg, dict) else arg
    yf = vector_apply(f, x, *arg)

    if normed_pdf:
        normed_factor = sum(yf) # assume equal binwidth
        yf /= normed_factor
    yf *= scale

    plt.plot(x, yf, **kwds)
    return x, yf


#draw comprison between function given args and data
def draw_compare_hist(f, arg, data, bins=100, bound=None, weights=None,
                      normed=False, use_w2=False, parts=False):
    """
    draw histogram of data with poisson error bar and f(x,*arg).

    ::

        np.rannd(10000)
        f = gaussian
        draw_compare_hist(f, {'mean':0,'sigma':1}, data, normed=True)

    **Arguments**

        - **f**
        - **arg** argument pass to f. Can be dictionary or list.
        - **data** data array
        - **bins** number of bins. Default 100.
        - **bound** optional boundary of plot in tuple form. If `None` is
          given, the bound is determined from min and max of the data. Default
          `None`
        - **weights** weights array. Default None.
        - **normed** optional normalized data flag. Default False.
        - **use_w2** scaled error down to the original statistics instead of
          weighted statistics.
        - **parts** draw parts of pdf. (Works with AddPdf and Add2PdfNorm).
          Default False.
    """
    bound = minmax(data) if bound is None else bound
    h, e = np.histogram(data, bins=bins, range=bound, weights=weights)
    err = None
    if weights is not None and use_w2:
        err, _ = np.histogram(data, bins=bins, range=bound,
                            weights=weights*weights)
        err = np.sqrt(err)
    else:
        err = np.sqrt(h)
    return draw_compare(f, arg, e, h, err, normed, parts)
