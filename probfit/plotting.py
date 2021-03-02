# -*- coding: utf-8 -*-
# Plotting is on python since this will make it much easier to debug and adjust
# no need to recompile everytime i change graph color....
# needs a serious refactor
from math import sqrt, ceil, floor
from warnings import warn
import numpy as np
from .nputil import mid, minmax, vector_apply
from .util import parse_arg, describe


def draw_simultaneous(self, minuit=None, args=None, errors=None, **kwds):
    from matplotlib import pyplot as plt
    numf = len(self.allf)
    ret = []
    numraw = sqrt(numf)
    numcol = ceil(numraw)
    numrow = floor(numraw) if floor(numraw) * numcol >= numf else ceil(numraw)

    for i in range(numf):
        plt.subplot(numrow, numcol, i + 1)
        part_args, part_errors = self.args_and_error_for(i, minuit, args, errors)
        ret.append(self.allf[i].draw(args=part_args, errors=part_errors, **kwds))

    return ret


def _get_args_and_errors(self, minuit=None, args=None, errors=None):
    """
    consistent algorithm to get argument and errors
    1) get it from minuit if minuit is available
    2) if not get it from args and errors
    2.1) if args is dict parse it.
    3) if all else fail get it from self.last_arg
    """
    ret_arg = None
    ret_error = None
    if minuit is not None:  # case 1
        ret_arg = minuit.args
        ret_error = minuit.errors
        return ret_arg, ret_error

    # no minuit specified use args and errors
    if args is not None:
        if isinstance(args, dict):
            ret_arg = parse_arg(self, args)
        else:
            ret_arg = args
    else:  # case 3
        ret_arg = self.last_arg

    if errors is not None:
        ret_error = errors

    return ret_arg, ret_error


def _param_text(parameters, arg, error):
    txt = u''
    for (k, v) in zip(parameters, arg):
        txt += u'%s = %5.4g' % (k, v)
        if error is not None:
            txt += u'±%5.4g' % error[k]
        txt += u'\n'
    return txt


    # from UML


def draw_ulh(self, minuit=None, bins=100, ax=None, bound=None,
             parmloc=(0.05, 0.95), nfbins=200, print_par=True, grid=True,
             args=None, errors=None, parts=False, show_errbars='normal',
             no_plot=False):
    from matplotlib import pyplot as plt
    error_ret = None
    part_ret = []

    ax = plt.gca() if ax is None and not no_plot else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    n, e = np.histogram(self.data, bins=bins, range=bound, weights=self.weights)
    dataint = (n * np.diff(e)).sum()
    data_ret = (e, n)

    if not show_errbars:
        if not no_plot:
            ax.hist(mid(e), bins=e, weights=n, histtype='step')
        error_ret = (np.sqrt(n), np.sqrt(n))
    else:
        w2 = None
        if show_errbars == 'normal':
            w2 = n
            error_ret = (np.sqrt(n), np.sqrt(n))
        elif show_errbars == 'sumw2':
            weights = None
            if self.weights != None:
                weights = self.weights ** 2
            w2, e = np.histogram(self.data, bins=e, weights=weights)
            error_ret = (np.sqrt(w2), np.sqrt(w2))
        else:
            raise ValueError('show_errbars must be \'normal\' or \'sumw2\'')
        if not no_plot:
            ax.errorbar(mid(e), n, np.sqrt(w2), fmt='b.', capsize=0,
                        zorder=0)

    # bound = (e[0], e[-1])
    draw_arg = [('lw', 2), ('zorder', 2)]
    if not parts:
        draw_arg.append(('color', 'r'))

    # Draw pdf with finer bins
    ef = np.linspace(e[0], e[-1], nfbins + 1)
    scale = dataint if not self.extended else nfbins / float(bins)
    total_ret = draw_pdf_with_edges(self.f, arg, ef, ax=ax, density=not self.extended, scale=scale,
                                    **dict(draw_arg))

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                ret = draw_pdf_with_edges(p, arg, ef, ax=ax, scale=scale, density=not self.extended, no_plot=no_plot)
                part_ret.append(ret)
    if not no_plot:
        ax.grid(grid)

    txt = _param_text(describe(self), arg, error)
    if not no_plot and print_par:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)
    return (data_ret, error_ret, total_ret, part_ret)


def draw_residual(x, y, yerr, xerr,
                  show_errbars=True, ax=None,
                  zero_line=True, grid=True,
                  **kwargs):
    """Draw a residual plot on the axis.

    By default, if show_errbars if True, residuals are drawn as blue points
    with errorbars with no endcaps. If show_errbars is False, residuals are
    drawn as a bar graph with black bars.

    **Arguments**

        - **x** array of numbers, x-coordinates

        - **y** array of numbers, y-coordinates

        - **yerr** array of numbers, the uncertainty on the y-values

        - **xerr** array of numbers, the uncertainty on the x-values

        - **show_errbars** If True, draw the data as a bar plot, else as an
          errorbar plot

        - **ax** Optional matplotlib axis instance on which to draw the plot

        - **zero_line** If True, draw a red line at :math:`y = 0` along the
          full extent in :math:`x`

        - **grid** If True, draw gridlines

        - **kwargs** passed to ``ax.errorbar`` (if ``show_errbars`` is True) or
          ``ax.bar`` (if ``show_errbars`` if False)

    **Returns**

    The matplotlib axis instance the plot was drawn on.
    """
    from matplotlib import pyplot as plt

    ax = plt.gca() if ax is None else ax

    if show_errbars:
        plotopts = dict(fmt='b.', capsize=0)
        plotopts.update(kwargs)
        pp = ax.errorbar(x, y, yerr, xerr, zorder=0, **plotopts)
    else:
        plotopts = dict(color='k')
        plotopts.update(kwargs)
        pp = ax.bar(x - xerr, y, width=2*xerr, **plotopts)

    if zero_line:
        ax.plot([x[0] - xerr[0], x[-1] + xerr[-1]], [0, 0], 'r-', zorder=2)

    # Take the `grid` kwarg to mean 'add a grid if True'; if grid is False and
    # we called ax.grid(False) then any existing grid on ax would be turned off
    if grid:
        ax.grid(grid)

    return ax


def draw_residual_ulh(self, minuit=None, bins=100, ax=None, bound=None,
                      parmloc=(0.05, 0.95), print_par=False,
                      args=None, errors=None, errbar_algo='normal', norm=False,
                      **kwargs):
    from matplotlib import pyplot as plt

    ax = plt.gca() if ax is None else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    n, e = np.histogram(self.data, bins=bins, range=bound, weights=self.weights)
    dataint = (n * np.diff(e)).sum()
    scale = dataint if not self.extended else 1.0
    w2 = None
    if errbar_algo == 'normal':
        w2 = n
    elif errbar_algo == 'sumw2':
        weights = None
        if self.weights != None:
            weights = self.weights ** 2
        w2, e = np.histogram(self.data, bins=e, weights=weights)
    else:
        raise ValueError('errbar_algo must be \'normal\' or \'sumw2\'')
    yerr = np.sqrt(w2)

    arg = parse_arg(self.f, arg, 1) if isinstance(arg, dict) else arg
    yf = vector_apply(self.f, mid(e), *arg)
    yf *= (scale * np.diff(e) if self.extended else scale)
    n = n - yf
    if norm:
        sel = yerr > 0
        n[sel] /= yerr[sel]
        yerr = np.ones(len(yerr))

    ax = draw_residual(mid(e), n, yerr, np.diff(e)/2.0, ax=ax, **kwargs)

    txt = _param_text(describe(self), arg, error)
    if print_par:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)


        # from chi2 regression


def draw_x2(self, minuit=None, ax=None, parmloc=(0.05, 0.95), print_par=True,
            args=None, errors=None, grid=True, parts=False, nbins=None, no_plot=False):
    from matplotlib import pyplot as plt
    error_ret = None
    part_ret = []

    ax = plt.gca() if ax is None and not no_plot else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    x = self.x
    y = self.y
    data_err = self.error

    data_ret = x, y
    if data_err is None:
        if not no_plot: ax.plot(x, y, '+')
        error_ret = (np.ones(len(self.x)), np.ones(len(self.x)))
    else:
        if not no_plot: ax.errorbar(x, y, data_err, fmt='.', zorder=0)
        error_ret = (data_err, data_err)
    draw_arg = [('lw', 2), ('zorder', 2)]
    draw_arg.append(('color', 'r'))

    # Draw PDF curve(s)
    if nbins is not None:
        x = np.linspace(x[0],x[-1], nbins)

    total_ret = draw_pdf_with_midpoints(self.f, arg, x, no_plot=no_plot, ax=ax, **dict(draw_arg))

    if not no_plot:
        ax.grid(grid)

    txt = _param_text(describe(self), arg, error)

    chi2 = self(*arg)
    if self.ndof > 0:
        txt += u'chi2/ndof = %5.4g(%5.4g/%d)' % (chi2 / self.ndof, chi2, self.ndof)
    else:
        txt += u'chi2/ndof = (%5.4g/%d)' % (chi2, self.ndof)

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                tmp = draw_pdf_with_midpoints(p, arg, x, ax=ax, no_plot=no_plot, **dict(draw_arg))
                part_ret.append(tmp)

    if print_par and not no_plot:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)

    return (data_ret, error_ret, total_ret, part_ret)


def draw_x2_residual(self, minuit=None, ax=None, args=None, errors=None, grid=True,
                     norm=False):
    from matplotlib import pyplot as plt

    ax = plt.gca() if ax is None else ax

    arg, _ = _get_args_and_errors(self, minuit, args, errors)

    x = self.x
    y = self.y
    data_err = self.error
    f = self.f

    arg = parse_arg(f, arg, 1) if isinstance(arg, dict) else arg
    yf = vector_apply(f, x, *arg)

    yplot = y - yf
    eplot = data_err if data_err is not None else np.zeros(len(x))
    if norm:
        if data_err is None:
            warn(RuntimeWarning('No error on data points; cannot normalize to error'))
        else:
            yplot = yplot / data_err
            eplot = data_err / data_err
    ax.errorbar(x, yplot, eplot, fmt='b+')
    ax.grid(grid)


    # from binned chi2


def draw_bx2(self, minuit=None, parmloc=(0.05, 0.95), nfbins=500, ax=None,
             print_par=True, args=None, errors=None, parts=False, grid=True,
             no_plot=False):
    from matplotlib import pyplot as plt
    part_ret = []

    ax = plt.gca() if ax is None and not no_plot else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    m = mid(self.edges)

    if not no_plot:
        ax.errorbar(m, self.h, self.err, fmt='.', zorder=0)
    data_ret = (self.edges, self.h)
    error_ret = (self.err, self.err)

    bound = (self.edges[0], self.edges[-1])

    scale = nfbins / float(self.bins)  # scale back to bins

    draw_arg = [('lw', 2), ('zorder', 2)]

    if not parts:
        draw_arg.append(('color', 'r'))

    total_ret = draw_pdf(self.f, arg, bins=nfbins, bound=bound, ax=ax, density=False,
                         scale=scale, no_plot=no_plot, **dict(draw_arg))

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                tmp = draw_pdf(p, arg, bound=bound, bins=nfbins, ax=ax, density=False,
                               scale=scale, no_plot=no_plot)
                part_ret.append(tmp)

    if not no_plot:
        ax.grid(grid)

    txt = _param_text(describe(self), arg, error)

    chi2 = self(*arg)
    if self.ndof > 0:
        txt += u'chi2/ndof = %5.4g(%5.4g/%d)' % (chi2 / self.ndof, chi2, self.ndof)
    else:
        txt += u'chi2/ndof = (%5.4g/%d)' % (chi2, self.ndof)

    if print_par and not no_plot:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)

    return (data_ret, error_ret, total_ret, part_ret)


    # from binnedLH


def draw_blh(self, minuit=None, parmloc=(0.05, 0.95),
             nfbins=1000, ax=None, print_par=True, grid=True,
             args=None, errors=None, parts=False, no_plot=False):
    from matplotlib import pyplot as plt
    part_ret = []

    ax = plt.gca() if ax is None and not no_plot else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    m = mid(self.edges)

    if self.use_w2:
        err = np.sqrt(self.w2)
    else:
        err = np.sqrt(self.h)

    n = np.copy(self.h)
    dataint = (n * np.diff(self.edges)).sum()
    scale = dataint if not self.extended else 1.0

    if not no_plot:
        ax.errorbar(m, n, err, fmt='.', zorder=0)
    data_ret = (self.edges, n)
    error_ret = (err, err)

    draw_arg = [('lw', 2), ('zorder', 2)]
    if not parts:
        draw_arg.append(('color', 'r'))
    bound = (self.edges[0], self.edges[-1])

    # scale back to bins
    if self.extended:
        scale = nfbins / float(self.bins)
    total_ret = draw_pdf(self.f, arg, bins=nfbins, bound=bound, ax=ax, density=not self.extended,
                         scale=scale, no_plot=no_plot, **dict(draw_arg))
    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                tmp = draw_pdf(p, arg, bins=nfbins, bound=bound, ax=ax,
                               density=not self.extended, scale=scale, no_plot=no_plot)
                part_ret.append(tmp)
    if not no_plot:
        ax.grid(grid)

    txt = _param_text(describe(self), arg, error)

    if print_par and not no_plot:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)

    return (data_ret, error_ret, total_ret, part_ret)


def draw_residual_blh(self, minuit=None, parmloc=(0.05, 0.95),
                      ax=None, print_par=False, args=None, errors=None,
                      norm=False, **kwargs):
    from matplotlib import pyplot as plt

    ax = plt.gca() if ax is None else ax

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    m = mid(self.edges)

    if self.use_w2:
        err = np.sqrt(self.w2)
    else:
        err = np.sqrt(self.h)

    n = np.copy(self.h)
    dataint = (n * np.diff(self.edges)).sum()
    scale = dataint if not self.extended else 1.0

    arg = parse_arg(self.f, arg, 1) if isinstance(arg, dict) else arg
    yf = vector_apply(self.f, m, *arg)
    yf *= (scale * np.diff(self.edges) if self.extended else scale)
    n = n - yf
    if norm:
        sel = err > 0
        n[sel] /= err[sel]
        err = np.ones(len(err))

    ax = draw_residual(m, n, err, np.diff(self.edges)/2.0, ax=ax, **kwargs)

    txt = _param_text(describe(self), arg, error)

    if print_par:
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)


def draw_compare(f, arg, edges, data, errors=None, ax=None, grid=True, normed=False, parts=False):
    """
    TODO: this needs to be rewritten
    """
    from matplotlib import pyplot as plt

    # arg is either map or tuple
    ax = plt.gca() if ax is None else ax
    arg = parse_arg(f, arg, 1) if isinstance(arg, dict) else arg
    x = (edges[:-1] + edges[1:]) / 2.0
    bw = np.diff(edges)
    yf = vector_apply(f, x, *arg)
    total = np.sum(data)
    if normed:
        ax.errorbar(x, data / bw / total, errors / bw / total, fmt='.b',
                    zorder=0)
        ax.plot(x, yf, 'r', lw=2, zorder=2)
    else:
        ax.errorbar(x, data, errors, fmt='.b', zorder=0)
        ax.plot(x, yf * bw, 'r', lw=2, zorder=2)

    # now draw the parts
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
                ax.plot(x, tmpy * scale, lw=2, alpha=0.5)
    plt.grid(grid)
    return x, yf, data


def draw_normed_pdf(f, arg, bound, bins=100, scale=1.0, density=True, ax=None, **kwds):
    return draw_pdf(f, arg, bound, bins=100, scale=1.0, density=True,
                    normed_pdf=True, ax=ax, **kwds)


def draw_pdf(f, arg, bound, bins=100, scale=1.0, density=True,
             normed_pdf=False, ax=None, **kwds):
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
    return draw_pdf_with_edges(f, arg, edges, ax=ax, scale=scale, density=density,
                               normed_pdf=normed_pdf, **kwds)


def draw_pdf_with_edges(f, arg, edges, ax=None, scale=1.0, density=True,
                        normed_pdf=False, **kwds):
    x = (edges[:-1] + edges[1:]) / 2.0
    bw = np.diff(edges)
    scale *= bw if not density else 1.

    return draw_pdf_with_midpoints(f, arg, x, ax=ax, scale=scale,
                                   normed_pdf=normed_pdf, **kwds)


def draw_pdf_with_midpoints(f, arg, x, ax=None, scale=1.0, normed_pdf=False, no_plot=False, **kwds):
    from matplotlib import pyplot as plt

    ax = plt.gca() if ax is None and not no_plot else ax
    arg = parse_arg(f, arg, 1) if isinstance(arg, dict) else arg
    yf = vector_apply(f, x, *arg)

    if normed_pdf:
        normed_factor = sum(yf)  # assume equal binwidth
        yf /= normed_factor
    yf *= scale

    if not no_plot:
        ax.plot(x, yf, **kwds)
    return x, yf


    # draw comparison between function given args and data


def draw_compare_hist(f, arg, data, bins=100, bound=None, ax=None, weights=None,
                      normed=False, use_w2=False, parts=False, grid=True):
    """
    draw histogram of data with poisson error bar and f(x,*arg).

    ::

        data = np.random.rand(10000)
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
    from matplotlib import pyplot as plt

    ax = plt.gca() if ax is None else ax
    bound = minmax(data) if bound is None else bound
    h, e = np.histogram(data, bins=bins, range=bound, weights=weights)
    err = None
    if weights is not None and use_w2:
        err, _ = np.histogram(data, bins=bins, range=bound,
                              weights=weights * weights)
        err = np.sqrt(err)
    else:
        err = np.sqrt(h)
    return draw_compare(f, arg, e, h, err, ax=ax, grid=grid, normed=normed, parts=parts)
