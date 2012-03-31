cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp,pow,fabs,log
from dist_fit.util import FakeFuncCode
def mid(np.ndarray x):
    return (x[:-1]+x[1:])/2.0
def minmax(x):
    return (min(x),max(x))

def compute_nll(f,np.ndarray data,w,arg,double badvalue):
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

def integrate1d(f, int nint, np.ndarray[np.double_t] midpoints, np.ndarray[np.double_t] binwidth, tuple arg=None):
    cdef double ret = 0
    cdef double bw = 0
    cdef double mp =0
    cdef int i=0
    cdef double x=0
    cdef double mpi=0
    if arg is None: arg = tuple()
    #print midpoints
    for i in range(nint-1):#mid has 1 less
        bw = binwidth[i]
        mp = midpoints[i]
        x = f(mp,*arg)
        
        ret += x*bw
        #print mp,bw,mpi,x,ret
    return ret

cdef class UnbinnedML:
    cdef object vf
    cdef public object f
    cdef object weights
    cdef np.ndarray data
    cdef public object func_code
    cdef int data_len
    cdef double badvalue
    def __init__(self, f, data,weights=None,badvalue=-1000):
        #self.vf = np.vectorize(f)
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.weights = weights
        #only make copy when type mismatch
        if data.dtype!=np.float64:
            self.data = data.astype(np.float64)
        else:
            self.data = data
        self.data_len = len(data)
        self.badvalue = badvalue

    def __call__(self,*arg):
        return compute_nll(self.f,self.data,self.weights,arg,self.badvalue)


#fit a line with given function using minimizing chi2
class Chi2Regression:
    #cdef object vf
    # cdef public object f
    # cdef object weights
    # cdef np.ndarray data
    # cdef public object func_code
    # cdef int data_len
    # cdef double badvalue
    def __init__(self, f, data,error=None,weights=None,normed=True,badvalue=-1000):
        #self.vf = np.vectorize(f)
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.weights = weights
        self.data = data
        self.data_len = len(data)
        self.badvalue = badvalue

    def __call__(self,*arg):
        #move this to ipython
        pass


cdef class Normalize:
    cdef f
    cdef double norm_cache
    cdef tuple last_arg
    cdef int nint
    cdef np.ndarray midpoints
    cdef np.ndarray binwidth
    cdef public func_code
    cdef public func_defaults
    cdef int ndep
    def __init__(self,f,range,prmt=None,nint=1000,normx=None):
        """
        normx [optional array] adaptive step for dependent variable
        """
        self.f = f
        self.norm_cache= 1.
        self.last_arg = None
        self.nint = normx.size() if normx is not None else nint
        normx = normx if normx is not None else np.linspace(range[0],range[1],nint)
        if normx.dtype!=normx.dtype:
            normx = normx.astype(np.float64)
        #print range
        #print normx
        self.midpoints = mid(normx)
        #print self.midpoints
        self.binwidth = np.diff(normx)
        self.func_code = FakeFuncCode(f,prmt)
        self.ndep = 1#TODO make the code doesn't depend on this assumption
        self.func_defaults = None #make vectorize happy

    def __call__(self,*arg):
        #print arg
        cdef double n 
        cdef double x
        n = self._compute_normalization(*arg)
        x = self.f(*arg)
        return x/n

    def _compute_normalization(self,*arg):
        cdef tuple targ = arg[self.ndep:]
        if targ == self.last_arg:#cache hit
            #yah exact match for float since this is expected to be used
            #in vectorize which same value are passed over and over
            pass
        else:
            self.last_arg = targ
            self.norm_cache = integrate1d(self.f,self.nint,self.midpoints,self.binwidth,targ)
        return self.norm_cache


#just a function wrapper with a fake manipulable func_code
cdef class FakeFunc:
    cdef f
    cdef public func_code
    cdef public func_defaults
    def __init__(self,f,prmt=None):
        self.f = f
        self.func_code = FakeFuncCode(f,prmt)
        self.func_defaults = f.func_defaults
    def __call__(self,*arg):
        return self.f(*arg)
