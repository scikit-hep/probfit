cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp,pow,fabs,log
#from .util import FakeFuncCode

def mid(np.ndarray x):
    return (x[:-1]+x[1:])/2

def minmax(np.ndarray x):
    return min(x),max(x)

#for converting numpy array to double
def float2double(a):
    if a is None or a.dtype == np.float64:
        return a
    else:
        return a.astype(np.float64)

class MinimalFuncCode:
    def __init__(self,arg):
        self.co_varnames = tuple(arg)
        self.co_argcount = len(arg)
    def append(self,varname):
        tmp = list(self.co_varnames)
        tmp.append(varname)
        self.co_varnames = tuple(tmp)
        self.co_argcount = len(self.co_varnames)

#fake copying func_code with renaming parameters
#and docking off parameters from the front
class FakeFuncCode:
    def __init__(self,f,prmt=None,dock=0,append=None):
        #f can either be tuple or function object
        if hasattr(f,'func_code'):#copy function code
            for attr in dir(f.func_code):
                if '__' not in attr: 
                    #print attr, type(getattr(f.func_code,attr))
                    setattr(self,attr,getattr(f.func_code,attr))
            self.co_argcount-=dock
            self.co_varnames = self.co_varnames[dock:]
            if prmt is not None:#rename parameters from the front
                for i,p in enumerate(prmt):
                    self.co_varnames[i] = p
            if isinstance(append,str): append = [append]
            if append is not None:
                old_count = self.co_argcount
                self.co_argcount+=len(append)
                self.co_varnames=tuple(list(self.co_varnames[:old_count])+append+list(self.co_varnames[old_count:]))
        else:#build a really fake one from bare bone
            raise TypeError('f does not have func_code')

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
