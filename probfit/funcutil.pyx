#cython: embedsignature=True
import numpy as np
from .util import describe

def merge_func_code(*arg,prefix=None,skip_first=False):
    assert(prefix is None or len(prefix)==len(arg))
    all_arg = []

    for i,f in enumerate(arg):
        tmp = []
        first = skip_first
        for vn in describe(f):
            newv = vn
            if not first and prefix is not None:
                newv = prefix[i]+newv
            first = False
            tmp.append(newv)
        all_arg.append(tmp)

    #now merge it
    #do something smarter
    merge_arg = []
    for a in all_arg:
        for v in a:
            if v not in merge_arg:
                merge_arg.append(v)

    #build the map list of numpy int array
    pos = []
    for a in all_arg:
        tmp = []
        for v in a:
            tmp.append(merge_arg.index(v))
        pos.append(np.array(tmp,dtype=np.int))
    return MinimalFuncCode(merge_arg), pos

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
