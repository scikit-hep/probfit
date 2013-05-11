#cython: embedsignature=True
cimport cython
from cpython cimport PyFloat_AsDouble, PyTuple_GetItem, PyTuple_GetItem,\
                     PyObject, PyTuple_SetItem, PyTuple_SetItem,\
                     PyTuple_New, Py_INCREF, PyFloat_FromDouble

import numpy as np
cimport numpy as np
from warnings import warn
from probfit_warnings import SmallIntegralWarning,\
                             SharedExtendeeExtenderParameter
from _libstat cimport integrate1d_with_edges, _vector_apply,\
                      has_ana_integral, integrate1d
from funcutil import FakeFuncCode, merge_func_code, FakeFunc
from util import describe
from random import Random

cpdef tuple construct_arg(tuple arg, np.ndarray[np.int_t] fpos):
    cdef int size = fpos.shape[0]
    cdef int i, itmp
    cdef np.int_t* fposdata = <np.int_t*>fpos.data
    cdef tuple ret = PyTuple_New(size)
    cdef object tmpo
    for i in range(size):
        itmp = fposdata[i]
        tmpo = <object>PyTuple_GetItem(arg, itmp)
        Py_INCREF(tmpo)
        #Py_INCREF(tmpo) #first one for the case second one for the steal
        PyTuple_SetItem(ret, i, tmpo)
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
    Make convolution from supplied **f** and **g**. If your functions are
    analytically convolutable you will be better off implementing
    analytically. This functor is implemented using numerical integration with
    bound there are numerical issue that will, in most cases, lightly affect
    normalization.s

    Argument from **f** and **g** is automatically merge by name. For example,

    ::

        f = lambda x, a, b, c: a*x**2+b*x+c
        g = lambda x, a, sigma: gaussian(x,a,sigma)
        h = Convolve(f,g,gbound)
        describe(h)#['x','a','b','c','sigma']

        #h is equivalent to
        def h_equiv(x, a, b, c, sigma):
            return Integrate(f(x-t, a, b, c)*g(x, a, b, c), t_range=gbound)

    .. math::

        \\text{Convolve(f,g)}(t, arg \ldots) =
                \int_{\\tau \in \\text{gbound}}
                f(t-\\tau, arg \ldots)\\times g(t, arg\ldots) \, \mathrm{d}\\tau

    **Argument**

        - **f** Callable object, PDF.
        - **g** Resolution function.
        - **gbound** Bound of the resolution. Supplied something such that
          g(x,*arg) is near 0 at the edges. Current implementation is
          multiply reverse slide add straight up from the definition so this
          bound is important. Overbounding is recommended.
        - **nbins** Number of bins in multiply reverse slide add. Default(1000)

    .. note::

        You may be worried about normalization. By the property of convolution:

        .. math::

            \int_{\mathbf{R}^d}(f*g)(x) \, dx=\left(\int_{\mathbf{R}^d}f(x) \, dx\\right)\left(\int_{\mathbf{R}^d}g(x) \, dx\\right).

        This means if you convolute two normalized distributions you get back
        a normalized distribution. However, since we are doing numerical
        integration here we will be off by a little bit.

    """
    #TODO: make this use analytical integral
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
        self.f = f
        self.g = g

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
    Transformed given **f** into extended from.

    ::

        def f(x,mu,sigma):
            return gaussian(x,mu,sigma)
        g = Extended(f) #g is equivalent to N*f(x,mu,sigma)
        describe(g) #('x','mu','sigma','N')

        #g is equivalent to
        def g_equiv(x, mu, sigma, N):
            return N*f(x, mu, sigma)

    **Arguments**
        - **f** call object. PDF.
        - **extname** optional string. Name of the extended parameter. Default
          `'N'`

    """
    cdef f
    cdef public func_code
    cdef public func_defaults

    def __init__(self, f, extname='N'):
        self.f = f
        if extname in describe(f):
            raise ValueError('%s is already taken pick something else for extname')
        self.func_code = FakeFuncCode(f,append=extname)
        #print self.func_code.__dict__
        self.func_defaults=None

    def __call__(self, *arg):
        cdef double N = arg[-1]
        cdef double fval = self.f(*arg[:-1])
        return N*fval

    def integrate(self, tuple bound, int nint, *arg):
        cdef double N = arg[-1]
        cdef double ana = integrate1d(self.f, bound, nint, arg[:-1])
        return N*ana


cdef class AddPdf:
    """
    Directly add PDF without normalization nor factor.
    Parameters are merged by names.

    ::

        def f(x, a, b, c):
            return do_something(x,a,b,c)
        def g(x, d, a, e):
            return do_something_else(x, d, a, e)

        h = AddPdf(f, g)# you can do AddPdf(f, g, h)
        #h is equivalent to
        def h_equiv(x, a, b, c, d, e):
            return f(x, a, b, c) + g(x, d, a, e)

    **Arguments**

        - **prefix** array of prefix string with length equal to number of
          callable object passed. This allows you to add two PDF without having
          parameters from the two merge. ::

                h2 = AddPdf(f, g, prefix=['f','g'])
                #h is equivalent to
                def h2_equiv(x, f_a, f_b, f_c, g_d, g_a, g_e):
                    return f(x, f_a, f_b, f_c) + g(x, g_d, g_a, g_e)

        - **factors** list of callable factor function if given Add pdf
          will simulate the pdf of the form::

                factor[0]*f + factor[1]*g

          Note that all argument for callable factors will be prefixed (if 
          given) as opposed to skipping the first one for pdf list. If None
          is given, all factors are assume to be constant 1. Default None.

        - **skip_prefix** list of variable that should not be prefixed.
          Default None. This is useful when you want to mix prefixing
          and sharing some of variable.

    """
    #FIXME: cache each part if called with same parameter
    cdef public object func_code
    cdef public object func_defaults
    cdef int arglen
    cdef list allpos
    cdef list factpos

    cdef tuple allf
    cdef tuple factors

    cdef readonly int numf

    cdef np.ndarray cache
    cdef np.ndarray factor_cache
    cdef list argcache
    cdef list factor_argcache

    cdef public int hit

    cdef list allfactors

    def __init__(self, *arg, prefix=None, factors=None, skip_prefix=None):
        if factors is not None and len(factors)!=len(arg):
            raise ValueError('factor is specified but has different length'
                             ' from arg.')
        allf = list(arg)
        if factors is not None:
            allf += factors

        self.func_code, allpos = merge_func_code(*arg, prefix=prefix, 
                                                 skip_first=True, 
                                                 factor_list=factors,
                                                 skip_prefix=skip_prefix)

        funcpos = allpos[:len(arg)]
        factpos = allpos[len(arg):]

        self.func_defaults=None
        self.arglen = self.func_code.co_argcount
        self.allf = arg # f function
        self.factors = tuple(factors) if factors is not None else None# factor function
        self.allpos = allpos # position for f arg
        self.factpos = factpos # position for factor arg 
        self.numf = len(self.allf)
        self.argcache = [None]*self.numf
        self.factor_argcache = [None]*self.numf
        self.cache = np.zeros(self.numf)
        self.factor_cache = np.zeros(self.numf)
        self.hit = 0

    def __call__(self, *arg):
        cdef tuple this_arg
        cdef double ret = 0.
        cdef double tmp = 0.
        cdef double tmp_factor = 0.
        cdef int i
        cdef np.ndarray thispos
        for i in range(self.numf):
            thispos = self.allpos[i]
            this_arg = construct_arg(arg, thispos)

            if self.argcache[i] is not None and fast_tuple_equal(this_arg, self.argcache[i], 0):
                tmp = self.cache[i]
                self.hit+=1
            else:
                tmp = self.allf[i](*this_arg)
                self.argcache[i]=this_arg
                self.cache[i]=tmp

            if self.factors is not None: # calculate factor
                factor_arg = construct_arg(arg, self.factpos[i])
                if self.factor_argcache[i] is not None and fast_tuple_equal(factor_arg, self.factor_argcache[i], 0):
                    tmp_factor = self.factor_cache[i]
                    self.hit+=1
                else:
                    tmp_factor = self.factors[i](*factor_arg)
                    self.factor_argcache[i] = factor_arg
                    self.factor_cache[i] = tmp_factor

                ret += tmp_factor*tmp
            else:
                ret += tmp
        return ret

    def parts(self):
        return [self._part(i) for i in range(self.numf)]

    def _part(self, int findex):

        def tmp(*arg):
            thispos = self.allpos[findex]
            this_arg = construct_arg(arg, thispos)
            ret = self.allf[findex](*this_arg)
            if self.factors is not None:
                facpos = self.factpos[findex]
                facarg = construct_arg(arg, facpos)
                fac = self.factors[findex](*facarg)
                ret *= fac
            return ret

        tmp.__name__ = getattr(self.allf[findex],'__name__','unnamedpart')
        ret = FakeFunc(tmp)
        ret.func_code = self.func_code
        return ret

    def eval_parts(self,*arg):
        cdef tuple this_arg
        cdef double tmp = 0.
        cdef int i
        cdef list ref
        cdef np.ndarray thispos
        ret = list()
        for i in range(self.numf):
            tmp = self._part(i)(*arg)
            ret.append(tmp)
        return tuple(ret)

    def integrate(self, tuple bound, int nint, *arg):
        cdef int findex
        cdef tuple this_arg
        cdef double ret = 0.
        cdef double thisint = 0.
        cdef double fac = 0.
        cdef np.ndarray[np.int_t] fpos
        cdef np.ndarray[np.int_t] facpos

        for findex in range(self.numf):
            fpos = self.allpos[findex]

            #docking off x and shift due to no x in arg
            this_arg = construct_arg(arg, fpos[1:]-1)
            thisf = self.allf[findex]
            fac = 1.
        
            if self.factors is not None:
                facpos = self.factpos[findex]
                # -1 accounting for no dependent variable in this arg
                facarg = construct_arg(arg, facpos-1) 
                fac = self.factors[findex](*facarg)
            
            thisint = integrate1d(thisf, bound, nint, this_arg)
            ret += fac*thisint
        
        return ret

cdef class AddPdfNorm:
    """
    Add PDF with normalization factor. Parameters are merged by name.

    ::

        def f(x, a, b, c):
            return do_something(x, a, b, c)
        def g(x, d, a, e):
            return do_something_else(x, d, a, e)
        def p(x, b, a, c):
            return do_something_other_thing(x,b,a,c)
        h = Add2PdfNorm(f, g, p)

        #h is equivalent to
        def h_equiv(x, a, b, c, d, e, f_0, f_1):
            return f_0*f(x, a, b, c)+ \\
                   f_1*g(x, d, a, e)+
                   (1-f_0-f_1)*p(x, b, a, c)

    **Arguments**
        - **facname** optional list of factor name of length=. If None is given
          factor name is automatically chosen to be `f_0`, `f_1` etc.
          Default None.

        - **prefix** optional prefix list to prefix arguments of each function.
          Default None.

        - **skip_prefix** optional list of variable that prefix will not be
          applied to. Default None(empty).


    """
    #FIXME: cache each part if called with same parameter
    cdef public object func_code
    cdef public object func_defaults
    cdef int arglen
    cdef int normalarglen
    cdef tuple allf
    cdef readonly int numf
    #cdef f
    #cdef g
    cdef list allpos
    #cdef np.ndarray fpos
    #cdef np.ndarray gpos

    def __init__(self, *arg ,facname=None, prefix=None, skip_prefix=None):

        self.func_code, self.allpos = merge_func_code(*arg,
            prefix=prefix, skip_first=True, skip_prefix=skip_prefix)

        if facname is not None and len(facname)!=len(arg)-1:
            raise(RuntimeError('length of facname and arguments must satisfy len(facname)==len(arg)-1'))

        self.normalarglen = self.func_code.co_argcount
        #TODO check name collisions here

        if facname is None:
            #automatic naming
            facname = ['f_%d'%i for i in range(len(arg)-1)]

        for fname in facname:
            self.func_code.append(fname)

        self.arglen = self.func_code.co_argcount
        self.func_defaults = None
        self.allf = arg
        self.numf = len(arg)


    def __call__(self,*arg):
        cdef ret = 0.
        cdef tuple farg
        cdef double allfac = 0.
        cdef double fac = 0.
        cdef int findex = 0

        for findex in range(self.numf):
            if findex!=self.numf-1: #not the last one
                fac = arg[self.normalarglen+findex]
                allfac += fac
            else: #last one
                fac = 1-allfac
            farg = construct_arg(arg,self.allpos[findex])
            ret += fac*self.allf[findex](*farg)
        return ret

    def parts(self):
        return [self._part(i) for i in range(self.numf)]

    def _part(self, int findex):
        #FIXME make this faster. How does cython closure work?
        def tmp(*arg):
            if findex!=self.numf-1: #not the last one
                fac = arg[self.normalarglen+findex]
            else: #last one
                fac = 1. - sum(arg[self.normalarglen:])
            thispos = self.allpos[findex]
            this_arg = construct_arg(arg, thispos)
            return fac*self.allf[findex](*this_arg)

        tmp.__name__ = getattr(self.allf[findex],'__name__','unnamedpart')
        ret = FakeFunc(tmp)
        ret.func_code = self.func_code
        return ret

    def eval_parts(self,*arg):

        cdef tuple farg
        cdef double allfac = 0.
        cdef double fac = 0.
        cdef int findex = 0
        cdef list ret = []

        for findex in range(self.numf):
            if findex!=self.numf-1: #not the last one
                fac = arg[self.normalarglen+findex]
                allfac += fac
            else: #last one
                fac = 1-allfac
            farg = construct_arg(arg,self.allpos[findex])
            ret.append(fac*self.allf[findex](*farg))
        return tuple(ret)

    def integrate(self, tuple bound, int nint, *arg):
        cdef int findex
        cdef double allfac = 0.
        cdef double fac = 0.
        cdef double thisint = 0.
        cdef double ret = 0.
        cdef np.ndarray[np.int_t] fpos
        for findex in range(self.numf):
            if findex!=self.numf-1: #not the last one
                # -1 since this arg has no x
                fac = arg[self.normalarglen+findex-1]
                allfac += fac
            else: #last one
                fac = 1-allfac
            fpos = self.allpos[findex]
            # docking off x and shift due to no x in arg
            farg = construct_arg(arg, fpos[1:]-1)
            f = self.allf[findex]

            thisint = integrate1d(f, bound, nint, farg)
            ret += thisint*fac
        return ret


cdef class Normalized:
    """
    Transformed PDF in to a normalized version. The normalization factor is
    cached according the shape parameters(all arguments except the first one).

    ::

        def f(x, a, b, c):
            return do_something(x, a, b, c)
        g = Normalized(f, (0., 1.))
        #g is eqivalent to (shown here without cache)
        def g_equiv(x, a, b, c):
            return f(x, a, b, c)/Integrate(f(x, a, b, c), range=(0., 1.))

    **Arguments**
        - **f** function to normalized.
        - **bound** bound of the normalization.
        - **nint** optional number of pieces to integrate. Default 300.
        - **warnfloat** optinal number of times it should warn if integral
          of the given function is really small. This usually indicate
          you bound doesn't make sense with given parameters.

    .. note::
        Integration implemented here is just a simple trapezoid rule.
        You are welcome to implement something better and submit a pull
        request.

    .. warning::
        Never reused Normalized object with different parameters in the same pdf.

        ::

            #DO NOT DO THIS
            def f(x,y,z):
                do_something(x,y,z)
            pdf1 = Normalized(f,(0,1)) #h has it's own cache

            pdf2 = rename(pdf1,['x','a','b'])#don't do this

            totalpdf = Add2PdfNorm(pdf1,pdf2)

        The reason is that Normalized has excatly one cache value. Everytime
        it's called with different parameters the cache is invalidate and it
        will recompute the integration which takes a long time. For the
        example given above, when calling totalpdf, calling to `pdf2` will
        always invalidate `pdf1` cache causing it to recompute integration
        for every datapoint `x`. The fix is easy::

            #DO THIS INSTEAD
            def f(x,y,z):
                do_something(x,y,z)
            pdf1 = Normalized(f,(0,1)) #h has it's own cache

            pdf2_temp = Normalized(f,(0,1)) #own separate cache
            pdf2 = rename(pdf2_temp,['x','a','b'])

            totalpdf = Add2PdfNorm(pdf1,pdf2)

    """
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
    def __init__(self,f,bound,nint=300,warnfloat=1):
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
        self.func_code = FakeFuncCode(f)
        self.ndep = 1#TODO make the code doesn't depend on this assumption
        self.func_defaults = None #make vectorize happy
        self.warnfloat=warnfloat
        self.floatwarned=0
        self.hit=0

    def __call__(self,*arg):
        #print arg
        cdef double n
        cdef double x
        n = self._compute_normalization(arg[self.ndep:])
        x = self.f(*arg)
        if self.floatwarned < self.warnfloat  and n < 1e-100:
            warn(SmallIntegralWarning(str(arg)))
            self.floatwarned+=1
        return x/n

    cpdef _compute_normalization(self, tuple arg):
        cdef tuple targ = arg
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

    def integrate(self, tuple bound, int bint, *arg):
        n = self._compute_normalization(arg)
        X = integrate1d(self.f, bound, bint, arg)
        if self.floatwarned < self.warnfloat  and n < 1e-100:
            warn(SmallIntegralWarning(str(arg)))
            self.floatwarned+=1
        return X/n

cdef class BlindFunc:
    """
    Transform a given parameter in the given **f** by a random shift
    so that the analyst won't see the true fitted value.
    
    .. math::
        BlindFunc(f, 'y', '123')(x, y , z) = f(x, y\pm \delta, z)

    ::

        def f(x,mu,sigma):
            return gaussian(x,mu,sigma)
        g= BlindFunc(f, toblind='mu', seedstring= 'abcxyz', width=1, signflip=True)
        describe(g) # ['x', 'mu', 'sigma']

    **Arguments**
        - **f** call object. A function or PDF.
        - **toblind** the name of parameter to be blinded
        - **seedstring** a string random number seed to control the random shift
        - **width** a Gaussian width that controls the random shift
        - **signflip** if True, sign of the parameter may be flipped
          before being shifted.

    """

    cdef f
    cdef public func_code
    cdef public func_defaults
    cdef int signflip
    cdef int argpos
    cdef double shift
    cdef char* toblind

    def __init__(self, f, toblind, seedstring, width=1, signflip=True):
        self.f = f
        if toblind not in describe(f):
            raise ValueError('%s is not in a recognized parameter'%toblind)
        self.func_code = FakeFuncCode(f)
        self.func_defaults = None

        mystery ='ambpel4.b4G#4hwW%&eNrw56wJE56N%wwgwywJj%whw'
        rnd1 = Random(seedstring)
        seed2 = list(seedstring+mystery)
        rnd1.shuffle(seed2, rnd1.random)
        seed2 = ''.join(seed2)
        myRandom = Random(seed2)

        self.signflip = myRandom.choice([-1,1])
        self.shift = myRandom.gauss(0, width)
        self.toblind = toblind
        self.argpos = describe(f).index(toblind)

    cpdef tuple __shift_arg__(self, tuple arg):
        cdef int numarg = len(arg)
        cdef tuple ret = PyTuple_New(numarg)
        cdef int i
        cdef object tmp, tmp2
        cdef double ftmp
        for i in range(numarg):
            if i!=self.argpos:
                tmp =  <object>PyTuple_GetItem(arg, i)
                Py_INCREF(tmp) # get is borrow and set is steal
                               # but <object> comes with inc ref + dec ref
                PyTuple_SetItem(ret, i, tmp)
            else:
                tmp = <object>PyTuple_GetItem(arg, i)
                ftmp = tmp
                ftmp = ftmp*self.signflip + self.shift
                tmp2 = PyFloat_FromDouble(ftmp)
                Py_INCREF(tmp2)
                PyTuple_SetItem(ret, i, tmp2)
        return ret

    def __call__(self, *arg):
        cdef tuple newarg = self.__shift_arg__(arg)
        return self.f(*newarg)

    def integrate(self, tuple bound, int nint, *arg):
        cdef tuple newarg = self.__shift_arg__(arg)
        return integrate1d(self.f, bound, nint, newarg)
