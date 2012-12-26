#cython: embedsignature=True
cimport cython
from cpython cimport PyFloat_AsDouble, PyTuple_GetItem, PyTuple_GET_ITEM,\
                     PyObject, PyTuple_SetItem, PyTuple_SET_ITEM,\
                     PyTuple_New, Py_INCREF

import numpy as np
cimport numpy as np

from _libstat cimport integrate1d_with_edges, _vector_apply
from funcutil import FakeFuncCode, merge_func_code
from util import describe


cpdef tuple construct_arg(tuple arg, np.ndarray[np.int_t] fpos):
    cdef int size = fpos.shape[0]
    cdef int i,itmp
    cdef np.int_t* fposdata = <np.int_t*>fpos.data
    cdef tuple ret = PyTuple_New(size)
    cdef object tmpo
    for i in range(size):
        itmp = fposdata[i]
        tmpo = <object>PyTuple_GET_ITEM(arg, itmp)
        Py_INCREF(tmpo)
        #Py_INCREF(tmpo) #first one for the case second one for the steal
        PyTuple_SET_ITEM(ret, i, tmpo)
    return ret

#TODO: optimize this may be check id() instead of actual comparison?
cpdef bint fast_tuple_equal(tuple t1, tuple t2 , int t2_offset) except *:
    #t=last_arg
    #t2=arg
    cdef double tmp1,tmp2
    cdef int i,ind
    cdef int tsize = len(t2)-t2_offset
    cdef bint ret = 0
    cdef double precision = 1e-16
    if len(t1) ==0 and tsize==0:
        return 1

    for i in range(tsize):
        ind = i+t2_offset
        tmp1 = PyFloat_AsDouble(<object>PyTuple_GetItem(t1,i))
        tmp2 =  PyFloat_AsDouble(<object>PyTuple_GetItem(t2,ind))
        ret = abs(tmp1-tmp2) < precision
        if not ret: break

    return ret

cdef class Convolve:#with gy cache
    """
    Convolve)
    """
    cdef int numbins
    cdef tuple gbound
    cdef double bw
    cdef int nbg
    cdef f #original f
    cdef g #original g
    cdef np.ndarray fpos#position of argument in f
    cdef np.ndarray gpos#position of argument in g
    cdef public object func_code
    cdef public object func_defaults
    cdef tuple last_garg
    cdef np.ndarray gy_cache

    #g is resolution function gbound need to be set so that the end of g is zero
    def __init__(self, f, g, gbound, nbins=1000):
        self.set_gbound(gbound,nbins)
        self.func_code, [self.fpos, self.gpos] = merge_func_code(f,g,skip_first=True)
        self.func_defaults = None

    def set_gbound(self,gbound,nbins):
        self.last_garg = None
        self.gbound,self.nbg = gbound,nbins
        self.bw = 1.0*(gbound[1]-gbound[0])/nbins
        self.gy_cache = None

    def __call__(self,*arg):
        #skip the first one
        cdef int iconv
        cdef double ret = 0
        cdef np.ndarray[np.double_t] gy,fy

        tmp_arg = arg[1:]
        x=arg[0]
        garg = list()

        for i in self.gpos: garg.append(arg[i])
        garg = tuple(garg[1:])#dock off the independent variable

        farg = list()
        for i in self.fpos: farg.append(arg[i])
        farg = tuple(farg[1:])

        xg = np.linspace(self.gbound[0],self.gbound[1],self.nbg)

        gy = None
        #calculate all the g needed
        if garg==self.last_garg:
            gy = self.gy_cache
        else:
            gy = _vector_apply(self.g, xg, garg)
            self.gy_cache=gy

        #now prepare f... f needs to be calculated and padded to f-gbound[1] to f+gbound[0]
        #yep this is not a typo because we are "reverse" sliding g onto f so we need to calculate f from
        # f-right bound of g to f+left bound of g
        fbound = x-self.gbound[1], x-self.gbound[0] #yes again it's not a typo
        xf = np.linspace(fbound[0], fbound[1], self.nbg)#yep nbg
        fy = _vector_apply(self.f, xf, farg)
        #print xf[:100]
        #print fy[:100]
        #now do the inverse slide g and f
        for iconv in range(self.nbg):
            ret += fy[iconv]*gy[self.nbg-iconv-1]

        #now normalize the integral
        ret*=self.bw

        return ret


cdef class Extended:
    """
    f = lambda x,y: x+y
    g = Extend(f) ==> g = lambda x,y,N: f(x,y)*N
    """
    cdef f
    cdef public func_code
    cdef public func_defaults
    def __init__(self,f,extname='N'):
        self.f = f
        if extname in describe(f):
            raise ValueError('%s is already taken pick something else for extname')
        self.func_code = FakeFuncCode(f,append=extname)
        #print self.func_code.__dict__
        self.func_defaults=None
    def __call__(self,*arg):
        cdef double N = arg[-1]
        cdef double fval = self.f(*arg[:-1])
        return N*fval

cdef class AddPdf:
    cdef public object func_code
    cdef public object func_defaults
    cdef int arglen
    cdef list allpos
    cdef tuple allf
    cdef int numf
    cdef np.ndarray cache
    cdef list argcache
    cdef public int hit
    cdef public int nparts
    def __init__(self,*arg,prefix=None):
        self.func_code, self.allpos = merge_func_code(*arg,prefix=prefix,skip_first=True)
        self.func_defaults=None
        self.arglen = self.func_code.co_argcount
        self.allf = arg
        self.numf = len(self.allf)
        self.argcache=[None]*self.numf
        self.cache = np.zeros(self.numf)
        self.hit = 0

    def __call__(self,*arg):
        cdef tuple this_arg
        cdef double ret = 0.
        cdef double tmp = 0.
        cdef int i
        cdef np.ndarray thispos
        for i in range(self.numf):
            thispos = self.allpos[i]
            this_arg = construct_arg(arg,thispos)
            if self.argcache[i] is not None and fast_tuple_equal(this_arg,self.argcache[i],0):
                tmp = self.cache[i]
                self.hit+=1
            else:
                tmp = self.allf[i](*this_arg)
                self.argcache[i]=this_arg
                self.cache[i]=tmp
            ret+=tmp
        return ret

    def eval_parts(self,*arg):
        cdef tuple this_arg
        cdef double tmp = 0.
        cdef int i
        cdef list ref
        cdef np.ndarray thispos
        ret = list()
        for i in range(self.numf):
            thispos = self.allpos[i]
            this_arg = construct_arg(arg,thispos)
            if self.argcache[i] is not None and fast_tuple_equal(this_arg,self.argcache[i],0):
                tmp = self.cache[i]
                self.hit+=1
            else:
                tmp = self.allf[i](*this_arg)
                self.argcache[i]=this_arg
                self.cache[i]=tmp
            ret.append(tmp)
        return tuple(ret)

cdef class Add2PdfNorm:
    cdef public object func_code
    cdef public object func_defaults
    cdef int arglen
    cdef f
    cdef g
    cdef np.ndarray fpos
    cdef np.ndarray gpos
    cdef np.ndarray farg_buffer
    cdef np.ndarray garg_buffer
    cdef public int nparts
    def __init__(self,f,g,facname='k_f'):
        self.func_code, [self.fpos, self.gpos] = merge_func_code(f,g,skip_first=True)
        self.func_code.append(facname)
        self.arglen = self.func_code.co_argcount
        self.func_defaults=None
        self.f=f
        self.g=g
        self.nparts = 2
        self.farg_buffer = np.empty(len(self.fpos))
        self.garg_buffer = np.empty(len(self.gpos))

    def __call__(self,*arg):
        cdef double fac = arg[-1]
        cdef tuple farg = construct_arg(arg,self.fpos)
        cdef tuple garg = construct_arg(arg,self.gpos)
        cdef double fv = self.f(*farg)
        cdef double gv = self.g(*garg)
        cdef double ret = fac*fv+(1.-fac)*gv
        return ret

    def eval_parts(self,*arg):
        cdef double fac = arg[-1]
        cdef tuple farg = construct_arg(arg,self.fpos)
        cdef tuple garg = construct_arg(arg,self.gpos)
        cdef double fv = fac*self.f(*farg)
        cdef double gv = (1.-fac)*self.g(*garg)
        return (fv,gv)

cdef class Normalized:
    cdef f
    cdef double norm_cache
    cdef tuple last_arg
    cdef int nint
    cdef np.ndarray edges
    #cdef np.ndarray binwidth
    cdef double binwidth
    cdef public object func_code
    cdef public object func_defaults
    cdef int ndep
    cdef int warnfloat
    cdef int floatwarned
    cdef public int hit
    def __init__(self,f,bound,prmt=None,nint=1000,warnfloat=1):
        self.f = f
        self.norm_cache= 1.
        self.last_arg = None
        self.nint = nint
        # normx = normx if normx is not None else np.linspace(range[0],range[1],nint)
        #         if normx.dtype!=normx.dtype:
        #             normx = normx.astype(np.float64)
        #print range
        #print normx
        self.edges = np.linspace(bound[0],bound[1],nint)
        #print self.midpoints
        self.binwidth = self.edges[1]-self.edges[0]
        self.func_code = FakeFuncCode(f,prmt)
        self.ndep = 1#TODO make the code doesn't depend on this assumption
        self.func_defaults = None #make vectorize happy
        self.warnfloat=warnfloat
        self.floatwarned=0
        self.hit=0

    def __call__(self,*arg):
        #print arg
        cdef double n
        cdef double x
        n = self._compute_normalization(*arg)
        x = self.f(*arg)
        if self.floatwarned < self.warnfloat  and n < 1e-100:
            print 'Potential float erorr:', arg
            self.floatwarned+=1
        return x/n

    def _compute_normalization(self, *arg):
        cdef tuple targ = arg[self.ndep:]
        #if targ == self.last_arg:#cache hit
        if self.last_arg is not None and fast_tuple_equal(targ,self.last_arg,0):
            #targ == self.last_arg:#cache hit
            #yah exact match for float since this is expected to be used
            #in vectorize which same value are passed over and over
            self.hit+=1
            pass
        else:
            self.last_arg = targ
            self.norm_cache = integrate1d_with_edges(self.f, self.edges,
                                                    self.binwidth, targ)
        return self.norm_cache

