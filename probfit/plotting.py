 # -*- coding: utf-8 -*-
#Plotting is on python since this will make it much easier to debug and adjsut
#no need to recompile everytime i change graph color....

#needs a serious refactor
from matplotlib import pyplot as plt
import numpy as np
from .nputil import mid, minmax, vector_apply
from util import parse_arg

#from UML
def draw_ulh(self, minuit=None, bins=100, ax=None, bound=None,
            parmloc=(0.05,0.95), nfbins=500, print_par=False):
    if ax is None: ax=plt.gca()
    arg = self.last_arg
    if minuit is not None: arg = minuit.args
    n, e, patches = ax.hist(self.data, bins=bins, weights=self.weights,
        histtype='step', range=bound, normed=True)
    m = mid(e)
    fxs = np.linspace(e[0],e[-1],nfbins)
    # v = vf(fxs,*self.last_arg)
    # plt.plot(fxs,v,color='r')
    v = vector_apply(self.f, fxs, *arg)
    ax.plot(fxs, v, color='r')

    ax.grid(True)
    minu = minuit
    ax = plt.gca()
    if minu is not None:
        #build text
        txt = u'';
        for k,v  in minu.values.items():
            err = minu.errors[k]
            txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
        if print_par: print txt
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)


#from chi2 regression
def draw_x2(self, minuit=None, ax=None, parmloc=(0.05,0.95), print_par=False):
    ax = ax=plt.gca() if ax is None else ax
    arg = self.last_arg
    if minuit is not None: arg = minuit.args
    x=self.x
    y=self.y
    err = self.error
    expy = vector_apply(self.f, x, *arg)

    if err is None:
        plt.plot(x,y,'+')
    else:
        plt.errorbar(x,y,err,fmt='.')
    plt.plot(x,expy,'r-')
    ax = plt.gca()
    minu = minuit
    if minu is not None:
        #build text
        txt = u'';
        for k,v  in minu.values.items():
            err = minu.errors[k]
            txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
        if print_par: print txt
        chi2 = self(*self.last_arg)
        txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2/self.ndof,chi2,self.ndof)
        plt.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)


#from binned chi2
def draw_bx2(self, minuit=None, parmloc=(0.05,0.95),
            nfbins=1000, ax=None, print_par=False):
    if ax is None: ax = plt.gca()
    arg = self.last_arg if minuit is None else minuit.args
    m = mid(self.edges)
    ax.errorbar(m, self.h, self.err, fmt='.')
    #assume equal spacing
    #self.edges[0],self.edges[-1]
    bw = self.edges[1]-self.edges[0]
    xs = np.linspace(self.edges[0], self.edges[-1], nfbins)
    #bw = np.diff(xs)
    xs = mid(xs)
    expy = vector_apply(self.f, xs, *arg)*bw

    ax.plot(xs, expy, 'r-')

    minu = minuit
    ax.grid(True)

    if minu is not None:
        #build text
        txt = u'';
        for k,v  in minu.values.items():
            err = minu.errors[k]
            txt += u'%s = %5.4g±%5.4g\n'%(k, v, err)
        chi2 = self(*self.last_arg)
        txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2/self.ndof, chi2, self.ndof)
        if print_par: print txt
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)


#from binnedLH
def draw_blh(self, minuit=None, parmloc=(0.05,0.95),
                nfbins=1000, ax = None, print_par=False):
    if ax is None: ax = plt.gca()
    arg = self.last_arg if minuit is None else minuit.args
    m = mid(self.edges)
    if self.use_w2:
        err = np.sqrt(self.w2)
    else:
        err = np.sqrt(self.h)

    if self.extended:
        ax.errorbar(m, self.h, err, fmt='.')
    else:
        scale = sum(self.h)
        ax.errorbar(m, self.h/scale, err/scale, fmt='.')

    #assume equal spacing
    #self.edges[0],self.edges[-1]
    bw = self.edges[1]-self.edges[0]
    xs = np.linspace(self.edges[0], self.edges[-1], nfbins)
    #bw = np.diff(xs)
    xs = mid(xs)
    expy = vector_apply(self.f, xs, *arg)*bw
    #if not self.extended: expy/=sum(expy)
    ax.plot(xs, expy, 'r-')

    minu = minuit
    ax.grid(True)

    if minu is not None:
        #build text
        txt = u'';
        sortk = minu.values.keys()
        sortk.sort()
        #for k,v  in minu.values.items():
        val = minu.values
        for k in sortk:
            v = val[k]
            err = minu.errors[k]
            txt += u'%s = %5.4g±%5.4g\n'%(k, v, err)
        #chi2 = self(*self.last_arg)
        #txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2,chi2*self.ndof,self.ndof)
        if print_par: print txt
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)


#useful for highlight some xrange
def vertical_highlight(x1, x2=None, color='g', alpha=0.3, ax=None):
    """

    :param x1:
    :param x2:
    :param color:
    :param alpha:
    :param ax:
    """
    from matplotlib import pyplot as plt
    if(ax is None): ax=plt.gca()
    if x2 is None:
        x1,x2=x1
    ylim = plt.gca().get_ylim()
    y = np.array(ylim)
    ax.fill_betweenx(y, np.array(x1,x1), np.array(x2,x2),
                    color=color, alpha=alpha)
    ax.set_ylim(ylim) #sometime it will decide to resize the plot


def draw_contour2d(fit, m, var1, var2, bins=12, bound1=None, bound2=None, lh=True):
    x1s, x2s, y = val_contour2d(fit, m, var1, var2, bins=bins,
                                bound1=bound1, bound2=bound2)
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
    yf = vector_apply(f, x, *arg)
    total = np.sum(data)
    if normed:
        plt.errorbar(x, data/bw/total, errors/bw/total, fmt='.b')
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
                plt.plot(x, tmpy*scale, lw=2, alpha=0.5)
    plt.grid(True)
    return x,yf,data


def draw_pdf(f, arg, bound, bins=100, scale=1.0, normed=True, **kwds):
    edges = np.linspace(bound[0], bound[1], bins)
    return draw_pdf_with_edges(f, arg, edges, scale=scale,
                                normed=normed, **kwds)

def draw_pdf_with_edges(f, arg, edges, scale=1.0, normed=True, **kwds):
    if isinstance(arg, dict): arg = parse_arg(f, arg, 1)
    x = (edges[:-1]+edges[1:])/2.0
    bw = np.diff(edges)
    yf = vector_apply(f, x, *arg)
    scale *= bw if not normed else 1.
    yf *= scale
    plt.plot(x, yf, lw=2, **kwds)
    return x,yf


#draw comprison between function given args and data
def draw_compare_hist(f, arg, data, bins=100, bound=None,weights=None,
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
    if bound is None: bound = minmax(data)
    h,e = np.histogram(data, bins=bins, range=bound, weights=weights)
    err = None
    if weights is not None and use_w2:
       err,_ = np.histogram(data, bins=bins, range=bound,
                            weights=weights*weights)
       err = np.sqrt(err)
    else:
        err = np.sqrt(h)
    return draw_compare(f, arg, e, h, err, normed, parts)


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
    if step is None: step = error / 20.

    xs = np.arange(bound[0], bound[1], step)
    ys = np.zeros(len(xs))
    for i, x in enumerate(xs):
        arg[pos] = x
        ys[i] = fit(*arg)

    return xs, ys
