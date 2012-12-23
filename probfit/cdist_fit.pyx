#cython: embedsignature=True
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp,pow,fabs,log,tgamma,lgamma,log1p,sqrt
from matplotlib import pyplot as plt
from .common import *
from warnings import warn
from cdist_func cimport *
import plotting

cdef extern from "math.h":
    bint isnan(double x)


cdef double compute_bin_lh_f(f,
                    np.ndarray[np.double_t] edges,
                    np.ndarray[np.double_t] h, #histogram,
                    np.ndarray[np.double_t] w2,
                    double N, #sum of h
                    tuple arg, double badvalue,
                    bint extend, bint use_sumw2) except *:
    """
    TODO:
    """
    cdef int i
    cdef int n = len(edges)

    cdef np.ndarray[np.double_t] fedges = cvectorize_f(f,edges,arg)
    cdef np.ndarray[np.double_t] midvalues = (fedges[1:]+fedges[:-1])/2
    cdef double ret = 0.
    cdef double bw = 0.
    cdef double E = cintegrate1d(f,(edges[0],edges[-1]),10000,arg)
    cdef double factor=0.
    cdef double th=0.
    cdef double tw=0.
    cdef double tm=0.
    for i in range(n-1):#h has length of n-1
        #ret -= h[i]*log(midvalues[i])#non zero subtraction
        bw = edges[i+1]-edges[i]
        th = h[i]
        tm = midvalues[i]
        if not extend:
            if not use_sumw2:
                ret -= cxlogyx(th,tm*N*bw)+(th-tm*bw*N)#h[i]*log(midvalues[i]/nh[i]) #subtracting h[i]*log(h[i]/(N*bw))
            else:
                if w2[i]<1e-200: continue
                tw = w2[i]
                tw = sqrt(tw)
                factor = th/tw
                ret -= factor*(cwlogyx(th,tm*N*bw,th)+(th-tm*bw*N))
        else:
            #print 'h',h[i],'midvalues',midvalues[i]*bw
            if not use_sumw2:
                ret -= cxlogyx(th,tm*bw)+(th-tm*bw)
            else:
                if w2[i]<1e-200: continue
                tw = w2[i]
                tw = sqrt(tw)
                factor = th/tw
                ret -= factor*(cwlogyx(th,tm*bw,th)+(th-tm*bw))
    return ret
    

cdef double compute_nll(f,np.ndarray data,w,arg,double badvalue) except *:
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
            thisdata = data_[i]
            lh = f(thisdata,*arg)
            if lh<=0:
                ret = badvalue
                break
            else:
                ret+=log(lh)*w_[i]
    return -1*ret


cdef double compute_chi2_f(f,
                    np.ndarray[np.double_t] x,
                    np.ndarray[np.double_t] y,
                    np.ndarray[np.double_t] error,
                    np.ndarray[np.double_t] weights,
                    tuple arg) except *:
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


cdef double compute_bin_chi2_f(f,
                np.ndarray[np.double_t] x,np.ndarray[np.double_t] y,
                np.ndarray[np.double_t]error,np.ndarray[np.double_t] binwidth,
                np.ndarray[np.double_t]weights,tuple arg) except *:
    cdef int usew = 1 if weights is not None else 0
    cdef int usee = 1 if error is not None else 0
    cdef int i
    cdef int datalen = len(x)
    cdef double diff
    cdef double fx
    cdef double ret=0
    cdef double err
    cdef double bw
    for i in range(datalen):
        bw = binwidth[i]
        fx = f(x[i],*arg)*bw
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


cdef class UnbinnedLH:
    cdef public object f
    cdef public object weights
    cdef public object func_code
    cdef public np.ndarray data
    cdef public int data_len
    cdef double badvalue
    cdef public tuple last_arg

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

    def draw(self, minuit=None, bins=100, ax=None, range=None,
            parmloc=(0.05,0.95),nfbins=500,print_par=False):
        return plotting.draw_ulh(self, minuit, bins, ax, range,
            parmloc, nfbins, print_par)

    def show(self,*arg,**kwd):
        self.draw(*arg,**kwd)
        plt.show()


#fit a line with given function using minimizing chi2
cdef class Chi2Regression:
    cdef public object f
    cdef public object weights
    cdef public object error
    cdef public object func_code
    cdef public int data_len
    cdef public double badvalue
    cdef public int ndof
    cdef public np.ndarray x
    cdef public np.ndarray y
    cdef public tuple last_arg


    def __init__(self, f, x, y, error=None, weights=None, badvalue=1000000):
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
        return compute_chi2_f(self.f,self.x,self.y,self.error,self.weights,arg)


    def draw(self, minuit=None, parmloc=(0.05,0.95), print_par=False):
        return plotting.draw_x2(self, minuit, parmloc, print_par)


    def show(self,*arg):
        self.draw(*arg)
        plt.show()


cdef class BinnedChi2:
    cdef public object f
    cdef public object vf
    cdef public object func_code
    cdef public np.ndarray h
    cdef public np.ndarray err
    cdef public np.ndarray edges
    cdef public np.ndarray midpoints
    cdef public np.ndarray binwidth
    cdef public int bins
    cdef public double mymin
    cdef public double mymax
    cdef public double badvalue
    cdef public tuple last_arg
    cdef public int ndof
    def __init__(self, f, data, bins=40, weights=None,range=None, sumw2=False,badvalue=1000000):
        self.f = f
        self.vf = np.vectorize(f)
        self.func_code = FakeFuncCode(f,dock=True)
        if range is None:
            range = minmax(data)
        self.mymin,self.mymax = range

        h,self.edges = np.histogram(data,bins,range=range,weights=weights)
        self.h = float2double(h)
        self.midpoints = mid(self.edges)
        self.binwidth = np.diff(self.edges)
        #sumw2 if requested
        if weights is not None and sumw2:
            w2 = weights*weights
            sw2,_ = np.histogram(data,bins,range=range,weights=w2)
            self.err = np.sqrt(sw2)
        else:
            self.err = np.sqrt(self.h)
        #check if error is too small
        if np.any(self.err<1e-5):
            raise ValueError('some bins are too small to do a chi2 fit. change your range')
        self.bins = bins
        self.badvalue = badvalue
        self.ndof = self.bins-(self.func_code.co_argcount-1)#fix this taking care of fixed parameter


    #lazy mid point implementation
    def __call__(self,*arg):
        self.last_arg = arg
        return compute_bin_chi2_f(self.f,self.midpoints,self.h,self.err,self.binwidth,None,arg)

    # def __call__(self,*arg):
    #        #can be optimized much better than this
    #        cdef np.ndarray[np.double_t] edges_values
    #        cdef np.ndarray[np.double_t] expy
    #        self.last_arg = arg
    #        edges_values = self.vf(self.edges,*arg)
    #        expy = mid(edges_values)
    #        return compute_chi2(self.h,expy,self.err)


    def draw(self, minuit=None, parmloc=(0.05,0.95),
                fbins=1000, ax = None, print_par=False):
        return plotting.draw_bx2(self, minuit,
            parmloc, fbins, ax, print_par)


    def show(self,*arg,**kwd):
        self.draw(*arg,**kwd)
        plt.show()


cdef class BinnedLH:
    cdef public object f
    cdef public object vf
    cdef public object func_code
    cdef public np.ndarray h
    cdef public np.ndarray w
    cdef public np.ndarray w2
    cdef public double N
    cdef public np.ndarray edges
    cdef public np.ndarray midpoints
    cdef public np.ndarray binwidth
    cdef public int bins
    cdef public double mymin
    cdef public double mymax
    cdef public double badvalue
    cdef public tuple last_arg
    cdef public int ndof
    cdef public bint extended
    cdef public bint use_w2
    def __init__(self, f, data, bins=40, weights=None, range=None, badvalue=1000000,
            extended=False, use_w2=False,use_normw=False):
        self.f = f
        self.vf = np.vectorize(f)
        self.func_code = FakeFuncCode(f,dock=True)
        self.use_w2 = use_w2
        self.extended = extended

        if range is None: range = minmax(data)
        self.mymin,self.mymax = range
        self.w = float2double(weights)
        if use_normw: self.w=self.w/np.sum(self.w)*len(self.w)
        h,self.edges = np.histogram(data,bins,range=range,weights=weights)
        self.h = float2double(h)
        self.N = csum(self.h)

        if weights is not None:
            self.w2,_ = np.histogram(data,bins,range=range,weights=weights*weights)
        else:
            self.w2,_ = np.histogram(data,bins,range=range,weights=None)
        self.w2 = float2double(self.w2)
        self.midpoints = mid(self.edges)
        self.binwidth = np.diff(self.edges)

        self.bins = bins
        self.badvalue = badvalue
        self.ndof = self.bins-(self.func_code.co_argcount-1)


    #lazy mid point implementation
    def __call__(self,*arg):
        self.last_arg = arg
        ret = compute_bin_lh_f(self.f,
                                self.edges,
                                self.h, #histogram,
                                self.w2,
                                self.N, #sum of h
                                arg, self.badvalue,
                                self.extended, self.use_w2)
        return ret


    def draw(self,minuit=None,parmloc=(0.05,0.95),fbins=1000,ax = None,print_par=False):
        return plotting.draw_blh(self, minuit, parmloc, fbins, ax, print_par)


    def show(self,*arg,**kwd):
        self.draw(*arg,**kwd)
        plt.show()

