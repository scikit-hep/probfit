cimport cython
from libc.math cimport exp,pow,fabs,log,sqrt
cdef double pi=3.1415926535897932384626433832795028841971693993751058209749445923078164062
import numpy as np
cimport numpy as np
from .common import *

cdef class Normalize

#peaking stuff
@cython.binding(True)
def ugaussian(double x, double mean, double sigma):
    """
    unnormed gaussian normalized for -inf to +inf
    """
    cdef double ret = 0
    if sigma < 1e-7:
        ret = 1e-300
    else:
        d = (x-mean)/sigma
        d2 = d*d
        ret = exp(-0.5*d2)
    return ret

@cython.binding(True)
def gaussian(double x, double mean, double sigma):
    """
    gaussian normalized for -inf to +inf
    """
    cdef double ret = 0
    if sigma < 1e-7:
        ret = 1e-300
    else:
        d = (x-mean)/sigma
        d2 = d*d
        ret = 1/(sqrt(2*pi)*sigma)*exp(-0.5*d2)
    return ret

@cython.binding(True)
def crystalball(double x,double alpha,double n,double mean,double sigma):
    """
    unnormalized crystal ball function 
    see http://en.wikipedia.org/wiki/Crystal_Ball_function
    """
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

#Background stuff
@cython.binding(True)
def argus(double x, double c, double chi):
    """
    unnormalized argus distribution
    see: http://en.wikipedia.org/wiki/ARGUS_distribution
    """
    cdef double xc = x/c
    cdef double xc2 = xc*xc
    cdef double ret = 0
    ret = xc/c*sqrt(1-xc2)*exp(-0.5*chi*chi*(1-xc2))
    return ret

#Polynomials
@cython.binding(True)
def linear(double x, double m, double c):
    """
    y=mx+c
    """
    cdef double ret = m*x+c
    return ret

@cython.binding(True)
def poly2(double x, double a, double b, double c):
    """
    y=ax^2+bx+c
    """
    cdef double ret = a*x*x+b*x+c
    return ret
@cython.binding(True)
def poly3(double x, double a, double b, double c, double d):
    """
    y=ax^2+bx+c
    """
    cdef double x2 = x*x
    cdef double x3 = x2*x
    cdef double ret = a*x3+b*x2+c*x+d
    return ret

cdef class Extend:
    """
    f = lambda x,y: x+y
    g = Extend(f) ==> g = lambda x,y,N: f(x,y)*N
    """
    cdef f
    cdef public func_code
    cdef public func_defaults
    def __init__(self,f,extname='N'):
        self.f = f
        if extname in f.func_code.co_varnames[:f.func_code.co_argcount]:
            raise ValueError('%s is already taken pick something else for extname')
        self.func_code = FakeFuncCode(f,append=extname)
        print self.func_code.__dict__
        self.func_defaults=None
    def __call__(self,*arg):
        cdef double N = arg[-1]
        cdef double fval = self.f(*arg[:-1])
        return N*fval

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

#to do runge kutta or something smarter
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