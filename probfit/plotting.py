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
            warn(RuntimeWarning('parts is set to True but function does '
                            'not have nparts or eval_parts method'))
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


