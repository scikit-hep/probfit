#cython: embedsignature=True
import numpy as np
from .util import describe

def rename(f, newarg):
    """
    Rename function parameters.

    ::

        def f(x,y,z):
            return do_something(x,y,z)
        g = rename(f, ['x','a','b'])
        print describe(g) # ['x','a','b']

        #g is equivalent to
        def g_equiv(x,a,b):
            return f(x,a,b)

    **Arguments**

        - **f** callable object
        - **newarg** list of new argument names

    **Returns**

        function with new argument.

    """
    return FakeFunc(f, newarg)

def merge_func_code(*arg, prefix=None, skip_first=False, factor_list=None,
                    skip_prefix=None): # this needs a seiours refactor
    """
    merge function arguments.::

        def f(x,y,z): return do_something(x,y,z)
        def g(x,z,p): return do_something(x,y,z)
        fc, pos = merge_func_code(f,g)
        #fc is now ('x','y','z','p')

    **Arguments**

        - **prefix** optional array of prefix string to add to each function
          argument. Default None.::

                def f(x,y,z): return do_something(x,y,z)
                def g(x,z,p): return do_something(x,y,z)
                fc, pos = merge_func_code(f,g,prefix=['f','g'], skip_first=True)
                #fc now ('x','f_y','f_z','f_p')

        - **skip_first** option boolean to skip prefixing the first argument
          of each function. This should normally be true when prefix is not
          None.

        - **factor_list** option list of functions. For function that skip_first
          should not be applied. The choice of prefix applice to factor_list
          will be in the same order as prefix. Default None

        - **skip_prefix** list of variable names that prefix should not be
          applied to.

    **Return**

        tuple(Merged Func Code, position array)

        position array **p[i][j]** indicates where in the argument list to
        obtain argument **j** for functin **i**. If fact

    .. seealso::

        functor.construct_arg
    """
    if prefix is not None and len(prefix)!=len(arg):
        raise ValueError('prefix should have the same length as number of ',
                         'functions. Expect %d(%r)' % (len(arg), arg))
    all_arg = []
    skip_prefix = set([]) if skip_prefix is None else set(skip_prefix)
    for i,f in enumerate(arg):
        tmp = []
        first = skip_first
        for vn in describe(f):
            newv = vn
            if not first and prefix is not None and newv not in skip_prefix:
                newv = prefix[i]+newv
            first = False
            tmp.append(newv)
        all_arg.append(tmp)

    if factor_list is not None:
        for i,f in enumerate(factor_list):
            tmp = []
            for vn in describe(f):
                newv = vn
                if prefix is not None and newv not in skip_prefix:
                    newv = prefix[i]+newv
                first = False
                tmp.append(newv)
            all_arg.append(tmp)

    #now merge it
    #FIXME: do something smarter
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
    cdef public __name__

    def __init__(self, f, prmt=None):
        self.f = f
        self.func_code = FakeFuncCode(f,prmt)
        self.func_defaults = getattr(f,'func_defaults',None)
        self.__name__ = getattr(f,'__name__','unnamed')

    def __call__(self, *arg):
        return self.f(*arg)
