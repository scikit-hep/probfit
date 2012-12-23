 # -*- coding: utf-8 -*-
#Plotting is on python since this will make it much easier to debug and adjsut
#no need to recompile everytime i change graph color....
from matplotlib import pyplot as plt
import numpy as np
from common import mid

#from UML
def draw_ulh(self,minuit=None,bins=100,ax=None,range=None,parmloc=(0.05,0.95),nfbins=500,print_par=False):
    if ax is None: ax=plt.gca()
    arg = self.last_arg
    if minuit is not None: arg = minuit.args
    n,e,patches = ax.hist(self.data,bins=bins,weights=self.weights,
        histtype='step',range=range,normed=True)
    m = mid(e)
    vf = np.vectorize(self.f)
    fxs = np.linspace(e[0],e[-1],nfbins)
    # v = vf(fxs,*self.last_arg)
    # plt.plot(fxs,v,color='r')
    v = vf(fxs,*arg)
    ax.plot(fxs,v,color='r')

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
        ax.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)


#from chi2 regression
def draw_x2(self,minuit=None,parmloc=(0.05,0.95),print_par=False):
    arg = self.last_arg
    if minuit is not None: arg = minuit.args
    vf = np.vectorize(self.f)
    x=self.x
    y=self.y
    err = self.error
    expy = vf(x,*arg)

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
def draw_bx2(self,minuit=None,parmloc=(0.05,0.95),fbins=1000,ax = None,print_par=False):
    if ax is None: ax = plt.gca()
    arg = self.last_arg
    if minuit is not None: arg = minuit.args
    m = mid(self.edges)
    ax.errorbar(m,self.h,self.err,fmt='.')
    #assume equal spacing
    #self.edges[0],self.edges[-1]
    bw = self.edges[1]-self.edges[0]
    xs = np.linspace(self.edges[0],self.edges[-1],fbins)
    #bw = np.diff(xs)
    xs = mid(xs)
    expy = self.vf(xs,*arg)*bw

    ax.plot(xs,expy,'r-')

    minu = minuit
    ax.grid(True)

    if minu is not None:
        #build text
        txt = u'';
        for k,v  in minu.values.items():
            err = minu.errors[k]
            txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
        chi2 = self(*self.last_arg)
        txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2/self.ndof,chi2,self.ndof)
        if print_par: print txt
        ax.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)


#from binnedLH
def draw_blh(self,minuit=None,parmloc=(0.05,0.95),fbins=1000,ax = None,print_par=False):
    if ax is None: ax = plt.gca()
    arg = self.last_arg
    if minuit is not None: arg = minuit.args
    m = mid(self.edges)
    if self.use_w2:
        err = np.sqrt(self.w2)
    else:
        err = np.sqrt(self.h)

    if self.extended:
        ax.errorbar(m,self.h,err,fmt='.')
    else:
        scale = sum(self.h)
        ax.errorbar(m,self.h/scale,err/scale,fmt='.')

    #assume equal spacing
    #self.edges[0],self.edges[-1]
    bw = self.edges[1]-self.edges[0]
    xs = np.linspace(self.edges[0],self.edges[-1],fbins)
    #bw = np.diff(xs)
    xs = mid(xs)
    expy = self.vf(xs,*arg)*bw
    #if not self.extended: expy/=sum(expy)
    ax.plot(xs,expy,'r-')

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
            txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
        #chi2 = self(*self.last_arg)
        #txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2,chi2*self.ndof,self.ndof)
        if print_par: print txt
        ax.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)
