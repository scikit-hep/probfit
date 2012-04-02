cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp,pow,fabs,log
from matplotlib import pyplot as plt
from .common import *
from cython.parallel import prange, parallel, threadid

cdef double compute_nll(f,np.ndarray data,w,arg,double badvalue):
    cdef int i=0
    cdef double lh=0
    cdef double nll=0
    cdef double ret=0
    cdef double thisdata=0
    cdef np.ndarray[np.double_t] data_ = data
    cdef int data_len = len(data)
    cdef np.ndarray[np.double_t] w_
    if w is None:
        for i in range(data_len):
        #for i in prange(data_len,nogil=True): #nope!!!! gilllllllll
            thisdata = data_[i]
            lh = f(thisdata,*arg)
            if lh<=0:
                ret = badvalue
                break
            else:
                ret+=log(lh)
    else:
        w_ = w
        for i in range(data_len):
        #for i in prange(data_len,nogil=True): #nope!!!! gillllllll
            thisdata = data_[i]
            lh = f(thisdata,*arg,nogil=True)
            if lh<=0:
                ret = badvalue
                break
            else:
                ret+=log(lh)*w_[i]        
    return -1*ret

cdef double compute_chi2_f(f,np.ndarray[np.double_t] x,np.ndarray[np.double_t] y ,
                np.ndarray[np.double_t]error,np.ndarray[np.double_t]weights,tuple arg):
    cdef int usew = 1 if weights is not None else 0
    cdef int usee = 1 if error is not None else 0
    cdef int i
    cdef int datalen = len(x)
    cdef double diff
    cdef double fx
    cdef double ret=0
    cdef double err
    for i in range(datalen):
        fx = f(x[i],*arg)
        diff = fx-y[i]
        if usee==1:
            err = error[i]
            if err<1e-10:
                raise ValueError('error contains value too small or negative')
            diff = diff/error[i]
        diff *= diff
        if usew==1:
            diff*=weights[i]
        ret += diff
    return ret

cdef double compute_chi2(np.ndarray[np.double_t] actual, np.ndarray[np.double_t] expected, np.ndarray[np.double_t] err):
    cdef int i=0
    cdef int maxi = len(actual)
    cdef double a
    cdef double e
    cdef double er
    cdef double ret
    for i in range(maxi):
        e = expected[i]
        a = actual[i]
        er = err[i]
        if er<1e-10:
            raise ValueError('error contains value too small or negative')
        ea = (e-a)/er
        ea *=ea
        ret +=ea
    return ret

cdef class UnbinnedML:
    cdef public object f
    cdef object weights

    cdef public object func_code
    cdef np.ndarray data
    cdef int data_len
    cdef double badvalue
    cdef tuple last_arg
    def __init__(self, f, data ,weights=None,badvalue=-100000):
        #self.vf = np.vectorize(f)
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.weights = weights
        #only make copy when type mismatch
        self.data = float2double(data)
        self.data_len = len(data)
        self.badvalue = badvalue

    def __call__(self,*arg):
        self.last_arg = arg
        return compute_nll(self.f,self.data,self.weights,arg,self.badvalue)
    
    @cython.binding(True)
    def draw(self,minuit=None,bins=100,ax=None,range=None,parmloc=(0.05,0.95),nfbins=500):
        if ax is None: ax=plt.gca()
        
        n,e,patches = ax.hist(self.data,bins=bins,weights=self.weights,
            histtype='step',range=range,normed=True)
        m = mid(e)
        vf = np.vectorize(self.f)
        fxs = np.linspace(e[0],e[-1],nfbins)
        # v = vf(fxs,*self.last_arg)
        # plt.plot(fxs,v,color='r')
        v = vf(fxs,*self.last_arg)
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
            print txt
            ax.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)
                
    def show(self,*arg):
        self.draw(*arg)
        plt.show()


#fit a line with given function using minimizing chi2
cdef class Chi2Regression:
    cdef public object f
    cdef object weights
    cdef object error
    cdef public object func_code
    cdef int data_len
    cdef double badvalue
    cdef int ndof
    cdef np.ndarray x
    cdef np.ndarray y
    cdef tuple last_arg
    
    def __init__(self, f, x, y,error=None,weights=None,badvalue=1000):
        #self.vf = np.vectorize(f)
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.weights = float2double(weights) 
        self.error = float2double(error)
        self.x = float2double(x)
        self.y = float2double(y)
        self.data_len = len(x)
        self.badvalue = badvalue
        self.ndof = self.data_len - (self.func_code.co_argcount-1)
        
    def __call__(self,*arg):
        self.last_arg = arg
        return compute_chi2_f(self.f,self.x,self.y,self.error,self.weights,arg)/self.ndof

    def draw(self,minuit=None,parmloc=(0.05,0.95)):
        vf = np.vectorize(self.f)
        x=self.x
        y=self.y
        err = self.error
        expy = vf(x,*self.last_arg)
        
        if err is None:
            plt.plot(x,y,',')
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
            print txt
            chi2 = self(*self.last_arg)
            txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2,chi2*self.ndof,self.ndof)
            plt.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)
  
    def show(self,*arg):
        self.draw(*arg)
        plt.show()

cdef class BinnedChi2:
    cdef public object f
    cdef public object vf
    cdef public object func_code
    cdef np.ndarray h
    cdef np.ndarray err
    cdef np.ndarray edges
    cdef np.ndarray midpoints
    cdef int bins
    cdef double mymin
    cdef double mymax
    cdef double badvalue
    cdef tuple last_arg
    cdef int ndof
    def __init__(self, f, data, bins=40, weights=None,range=None, sumw2=False,badvalue=-1000):
        self.f = f
        self.vf = np.vectorize(f)
        self.func_code = FakeFuncCode(f,dock=True)
        if range is None:
            range = minmax(data)
        self.mymin,self.mymax = range 
        
        h,self.edges = np.histogram(data,bins,range=range,weights=weights)
        self.h = float2double(h)
        self.midpoints = mid(self.edges)
        #sumw2 if requested
        if weights is not None and sumw2:
            w2 = weights*weights
            sumw2 = np.hist(data,bins,range=range,weights=w2)
            self.err = np.sqrt(sumw2)
        else:
            self.err = np.sqrt(self.h)
        #check if error is too small
        if np.any(self.err<1e-5):
            raise ValueError('some bins are too small to do a chi2 fit. change your range')
        self.bins = bins
        self.badvalue = badvalue
        self.ndof = self.bins-(self.func_code.co_argcount-1)
    
    #lazy mid point implementation
    def __call__(self,*arg):
        self.last_arg = arg
        return compute_chi2_f(self.f,self.midpoints,self.h,self.err,None,arg)
    
    # def __call__(self,*arg):
    #        #can be optimized much better than this
    #        cdef np.ndarray[np.double_t] edges_values
    #        cdef np.ndarray[np.double_t] expy
    #        self.last_arg = arg
    #        edges_values = self.vf(self.edges,*arg)
    #        expy = mid(edges_values)
    #        return compute_chi2(self.h,expy,self.err)

    def draw(self,minuit=None,parmloc=(0.05,0.95)):
        m = mid(self.edges)
        plt.errorbar(m,self.h,self.err,fmt='.')
        
        expy = self.vf(self.edges,*self.last_arg)
        expy = mid(expy)
        
        plt.plot(m,expy,'r-')
        
        minu = minuit
        plt.grid(True)

        ax = plt.gca()
        if minu is not None:
            #build text
            txt = u'';
            for k,v  in minu.values.items():
                err = minu.errors[k]
                txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
            chi2 = self(*self.last_arg)
            txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2,chi2*self.ndof,self.ndof)
            print txt
            plt.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)
    
    def show(self,*arg):
        self.draw(*arg)
        plt.show()
    