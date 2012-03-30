# cython: binding=True
# mode: run
# tag: cyfunction
from libc.math cimport exp,pow,fabs,log
import numpy as np
cimport numpy as np
import inspect

def mid(x):
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
            if prmt is not None:#rename parameters
                assert(len(prmt)==f.func_code.co_argcount)
                for i in range(f.func_code.co_argcount):
                    self.co_varnames[i] = prmt[i]
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
        self.vf = np.vectorize(f)
        self._norm_cache= 1.
        self._last_arg = None
        self._norm = 1
        self._normx = normx if normx is not None else np.linspace(range[0],range[1],nint)
        self._midpoints = mid(self._normx)
        self._binwidth = np.diff(self._normx)
        self.func_code = FakeFuncCode(f,prmt)
        self._ndep = 1#TODO make the code doesn't depend on this assumption
        self.func_defaults = None #make vectorize happy
    def __call__(self,*arg):
        self._compute_normalization(*arg)
        return self.f(*arg)/self._norm_cache
    def _compute_normalization(self,*arg):
        if arg[self._ndep:] == self._last_arg:#cache hit
            #yah exact match for float since this is expected to be used
            #in vectorize which same value are passed over and over
            pass
        else:
            self._last_arg = arg[self._ndep:]
            v = self.vf(self._midpoints,*(self._last_arg))
            #print self._midpoints
            self._norm_cache = np.sum(v*self._binwidth)
        return self._norm_cache

def cball(double x,double alpha,double n,double mean,double sigma):
        cdef double d = (x-mean)/sigma
        cdef double ret = 0
        cdef double A = 0
        cdef double B = 0
        
        if d > -alpha : 
            ret = exp(-0.5*d**2)
        else:
            al = fabs(alpha)
            A=pow(n/al,n)*exp(-al**2/2.)
            B=n/al-al
            ret = A*pow(B-d,-n)
        return ret
#and a decorator so people can do
#@normalized_function(xmin,xmax)
#def function_i_have_no_idea_how_to_normalize(x,y,z)
#   return complicated_function(x,y,z)
class normalized_function:
    def __init__(self,xmin,xmax):
        self.xmin = xmin
        self.xmax = xmax
    def __call__(self,f):
        return Normalize(f,(self.xmin,self.xmax))
