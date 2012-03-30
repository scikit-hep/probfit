cimport cython
import numpy as np
cimport numpy as np
import inspect
from libc.math cimport exp,pow,fabs,log

def mid(np.ndarray x):
    return (x[:-1]+x[1:])/2.0

#fake copying func_code with renaming parameters
#and docking off parameters from the front
class FakeFuncCode:
    def __init__(self,f,prmt=None,dock=0):
        #f can either be tuple or function object
        if hasattr(f,'func_code'):#copy function code
            for attr in dir(f.func_code):
                if '__' not in attr: 
                    setattr(self,attr,getattr(f.func_code,attr))
            if prmt is not None:#rename parameters from the front
                for i,p in enumerate(prmt):
                    self.co_varnames[i] = p
        else:#build a really fake one from bare bone
            self.co_argcount = len(prmt)
            self.co_varnames = prmt
            #make the rest missing on purpose so it throws error
            #ans we will know what we want
    def dock(self):
        """remove the first variable from func_code"""
        self.co_argcount-=1
        self.co_varnames=self.co_varnames[1:]

#to do make it work for more than 1 dependent variable

class UnbinnedML:
    def __init__(self, f, data,weights=None,normed=True):
        self.vf = np.vectorize(f)
        self.f = f
        self.func_code=FakeFuncCode(f)
        self.weights = weights
        #there is no need to dock off first arg because 
        #it simulates self argument for call
        self.data = data

    def __call__(self,*arg):
        
        lh = self.vf(self.data,*arg)
        w = self.weights
        if w is not None:
            lh*=w
        nll = -1*sum(np.log(lh))
        return nll
        
    def pdf(self,*arg):
        f_tmp = lambda x: self.f(x,*arg)
        return f_tmp
    
    def vpdf(self,*arg):
        return np.vectorize(self.pdf(*arg))

class Normalize:

    def __init__(self,f,range,prmt=None,nint=1000,normx=None):
        """
        normx [optional array] adaptive step for dependent variable
        """
        self.f = f

        self.norm_cache= 1.
        self.last_arg = None
        self.nint = normx.size() if normx is not None else nint
        normx = normx if normx is not None else np.linspace(range[0],range[1],nint)
        
        self.midpoints = mid(normx)
        self.binwidth = np.diff(normx)
        self.func_code = FakeFuncCode(f,prmt)
        self.ndep = 1#TODO make the code doesn't depend on this assumption
        self.func_defaults = None #make vectorize happy
    def __call__(self,*arg):
        cdef double x =0
        cdef double n =0
        self._compute_normalization(*arg)
        x = self.f(*arg)
        n = self.norm_cache
        return x/n
    def _compute_normalization(self,*arg):
        cdef tuple targ = arg[self.ndep:]
        cdef double integral=0
        if targ == self.last_arg:#cache hit
            #yah exact match for float since this is expected to be used
            #in vectorize which same value are passed over and over
            pass
        else:
            self.last_arg = targ
            self.norm_cache = _integrate1d(self.f,self.nint,self.midpoints,self.binwidth,targ)
        return self.norm_cache

def _integrate1d(f, int nint, np.ndarray[np.double_t] midpoints, np.ndarray[np.double_t] binwidth, tuple arg=None):
    cdef double ret = 0
    cdef double bw = 0
    cdef double mp =0
    cdef int i=0
    cdef double x=0
    cdef double mpi=0
    if arg is None: arg = tuple()
    for i in range(nint-1):#mid has 1 less
        bw = binwidth[i]
        mp = midpoints[i]
        x = f(mpi,*arg)
        ret += x/bw
    return ret

#just a function wrapper with a fake manipulable func_code
class FakeFunc:
    def __init__(self,f,prmt=None):
        self.f = f
        self.func_code = FakeFuncCode(f,prmt)
        self.func_defaults = f.func_defaults
    def __call__(self,*arg):
        return self.f(*arg)


#and a decorator so people can do
#@normalized_function(xmin,xmax)
#def function_i_have_no_idea_how_to_normalize(x,y,z)
#   return complicated_function(x,y,z)
#
cdef class normalized_function:
    cdef double xmin
    cdef double xmax
    def __init__(self,double xmin,double xmax):
        self.xmin = xmin
        self.xmax = xmax
    def __call__(self,f):
        return Normalize(f,(self.xmin,self.xmax))

cdef class rename_parameter:
    def __init__(self,*arg):
        self.arg = arg
    def __call__(self,f):
        return FakeFunc(f,self.arg)


