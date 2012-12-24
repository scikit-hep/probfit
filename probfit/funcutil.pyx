#cython: embedsignature=True
import numpy as np
from .util import describe

class MinimalFuncCode:
    def __init__(self, arg):
        self.co_varnames = tuple(arg)
        self.co_argcount = len(arg)


    def append(self, varname):
        tmp = list(self.co_varnames)
        tmp.append(varname)
        self.co_varnames = tuple(tmp)
        self.co_argcount = len(self.co_varnames)


#fake copying func_code with renaming parameters
#and docking off parameters from the front
class FakeFuncCode:
    def __init__(self, f, prmt=None, dock=0, append=None):
        #f can either be tuple or function object
        self.co_varnames=describe(f)
        self.co_argcount=len(self.co_varnames)
        self.co_argcount-=dock
        self.co_varnames = self.co_varnames[dock:]

        if prmt is not None:#rename parameters from the front
            for i,p in enumerate(prmt):
                self.co_varnames[i] = p

        if isinstance(append,str): append = [append]

        if append is not None:
            old_count = self.co_argcount
            self.co_argcount+=len(append)
            self.co_varnames = tuple(
                                    list(self.co_varnames[:old_count])+
                                    append+
                                    list(self.co_varnames[old_count:]))


cdef class FakeFunc:
    cdef f
    cdef public func_code
    cdef public func_defaults


    def __init__(self, f, prmt=None):
        self.f = f
        self.func_code = FakeFuncCode(f,prmt)
        self.func_defaults = f.func_defaults


    def __call__(self, *arg):
        return self.f(*arg)
