#cython: embedsignature=True
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, fabs, log, tgamma, lgamma, sqrt
include "log1p_patch.pxi"
from warnings import warn
from probfit_warnings import LogWarning
np.import_array()

cpdef np.ndarray[np.double_t] _vector_apply(f,np.ndarray[np.double_t] x,tuple arg):
    cdef int i
    cdef int n = len(x)
    cdef np.ndarray[np.double_t] ret = np.empty(n,dtype=np.double)#fast_empty(n)
    cdef double tmp
    for i in range(n):
        tmp = f(x[i],*arg)
        ret[i]=tmp
    return ret


cpdef double csum(np.ndarray x):
    cdef int i
    cdef np.ndarray[np.double_t] xd = x
    cdef int n = len(x)
    cdef double s=0.
    for i in range(n):
        s+=xd[i]
    return s

cpdef inline bint has_ana_integral(f):
    integrate = getattr3(f, 'integrate', None)
    return integrate is not None

#currently bin width must be uniform to save tons of multiplications
#based on simpson 3/8
cpdef double integrate1d_with_edges(f, np.ndarray edges, double bw, tuple arg) except *:
    if has_ana_integral(f):
        # this is not exactly correct for non-uniform bin
        return f.integrate((edges[0],edges[-1]), len(edges-1), *arg)
    return  simpson38(f, edges, bw, arg)

cdef double simpson38(f, np.ndarray edges, double bw, tuple arg):
    cdef np.ndarray[np.double_t] yedges = _vector_apply(f, edges, arg)
    cdef np.ndarray[np.double_t] left38 = _vector_apply(f, (2.*edges[1:]+edges[:-1])/3., arg)
    cdef np.ndarray[np.double_t] right38 = _vector_apply(f, (edges[1:]+2.*edges[:-1])/3., arg)
    return bw/8.*( csum(yedges)*2.+csum(left38+right38)*3. - (yedges[0]+yedges[-1]) ) #simpson3/8


#TODO: do something smarter like dynamic edge based on derivative or so
cpdef double integrate1d(f, tuple bound, int nint, tuple arg=None) except*:
    """
    compute 1d integral
    """
    if arg is None: arg = tuple()
    if has_ana_integral(f):
        return f.integrate(bound, nint, *arg)
    cdef double ret = 0
    cdef np.ndarray[np.double_t] edges = np.linspace(bound[0], bound[1], nint+1)
    #cdef np.ndarray[np.double_t] bw = edges[1:]-edges[:-1]
    cdef double bw = edges[1]-edges[0]
    return simpson38(f, edges, bw, arg)


#compute x*log(y/x) to a good precision especially when y~x
cpdef double xlogyx(double x,double y):
    cdef double ret
    if x<1e-100:
        warn(LogWarning('x is really small return 0'))
        return 0.
    if x<y:
        ret = x*log1p((y-x)/x)
    else:
        ret = -x*log1p((x-y)/y)
    return ret


#compute w*log(y/x) where w < x and goes to zero faster than x
cpdef double wlogyx(double w,double y, double x):
    if x<1e-100:
        warn(LogWarning('x is really small return 0'))
        return 0.
    if x<y:
        ret = w*log1p((y-x)/x)
    else:
        ret = -w*log1p((x-y)/y)
    return ret

#these are performance critical code
cpdef double compute_bin_lh_f(f,
                    np.ndarray[np.double_t] edges,
                    np.ndarray[np.double_t] h, #histogram,
                    np.ndarray[np.double_t] w2,
                    double N, #sum of h
                    tuple arg, double badvalue,
                    bint extend, bint use_sumw2, int nint_subdiv) except *:
    """
    Calculate binned likelihood. The behavior depends whether extended
    likelihood and whether use_sumw2 is requested

    - ``extended=False`` and ``use_sumw2=False``. The second term in the sum is
       to subtract off the minimum to get better precision.

      .. math::
            $-\sum_i h_i * log(f(x_i,...)*N*bin_width) - (h_i-f(x_i,...)*N*bin_width)$

    - ``extended=False`` and ``use_sumw2=True``

    - ``extended=True`` and ``use_sumw2=False``

    - ``extended=True`` and ``use_sumw2=True``
    """
    cdef int i
    cdef int n = len(edges)

    cdef double ret = 0.
    cdef double bw = 0.
    
    cdef double factor=0.
    cdef double th=0.
    cdef double tw=0.
    cdef double tm=0.
    for i in range(n-1):#h has length of n-1
        #ret -= h[i]*log(midvalues[i])#non zero subtraction
        bw = edges[i+1]-edges[i]
        th = h[i]
        tm = integrate1d(f, (edges[i],edges[i+1]), nint_subdiv, arg)
        if not extend:
            if not use_sumw2:
                ret -= xlogyx(th,tm*N)+(th-tm*N)
                #h[i]*log(midvalues[i]/nh[i]) #subtracting h[i]*log(h[i]/(N*bw))
                #the second term is added for added precision near the minimum
            else:
                if w2[i]<1e-200: continue
                tw = w2[i]
                #tw = sqrt(tw)
                factor = th/tw
                ret -= factor*(wlogyx(th,tm*N,th)+(th-tm*N))
        else:
            #print 'h',h[i],'midvalues',midvalues[i]*bw
            if not use_sumw2:
                ret -= xlogyx(th,tm)+(th-tm)
            else:
                if w2[i]<1e-200: continue
                tw = w2[i]
                #tw = sqrt(tw)
                factor = th/tw
                ret -= factor*(wlogyx(th,tm,th)+(th-tm))

    return ret


cpdef double compute_nll(f,np.ndarray data,w,arg,double badvalue) except *:
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


cpdef double compute_chi2_f(f,
                    np.ndarray[np.double_t] x,
                    np.ndarray[np.double_t] y,
                    np.ndarray[np.double_t] error,
                    np.ndarray[np.double_t] weights,
                    tuple arg) except *:
    cdef int usew = 1 if weights is not None else 0
    cdef int usee = 1 if error is not None else 0
    cdef int i
    cdef int datalen = len(x)
    cdef double diff = 0.
    cdef double fx = 0.
    cdef double ret = 0.
    cdef double err = 0.
    for i in range(datalen):
        fx = f(x[i],*arg)
        diff = fx-y[i]
        if usee==1:
            err = error[i]
            if err<1e-10:
                raise ValueError('error contains value too small or negative')
            diff = diff/error[i]
        diff *= diff
        if usew==1:
            diff*=weights[i]
        ret += diff
    return ret


cpdef double compute_bin_chi2_f(f,
                    np.ndarray[np.double_t] edges,
                    np.ndarray[np.double_t] y,
                    np.ndarray[np.double_t] error,
                    np.ndarray[np.double_t] weights,
                    tuple arg,
                    int nint_subdiv) except *:
    cdef int usew = 1 if weights is not None else 0
    cdef int usee = 1 if error is not None else 0
    cdef int i
    cdef int datalen = len(edges)-1 
    cdef double diff
    cdef double fx
    cdef double ret = 0.
    cdef double err
    cdef double bw

    for i in range(datalen):
        fx = integrate1d(f, (edges[i],edges[i+1]), nint_subdiv, arg)
        diff = fx-y[i]
        if usee==1:
            err = error[i]
            if err<1e-10:
                raise ValueError('error contains value too small or negative')
            diff = diff/error[i]
        diff *= diff
        if usew==1:
            diff*=weights[i]
        ret += diff
    return ret


cpdef double compute_chi2(np.ndarray[np.double_t] actual,
                        np.ndarray[np.double_t] expected,
                        np.ndarray[np.double_t] err) except *:
    cdef int i=0
    cdef int maxi = len(actual)
    cdef double a
    cdef double e
    cdef double er
    cdef double ret = 0.
    for i in range(maxi):
        e = expected[i]
        a = actual[i]
        er = err[i]
        if er<1e-10:
            raise ValueError('error contains value too small or negative')
        ea = (e-a)/er
        ea *= ea
        ret += ea
    return ret

cpdef compute_cdf(np.ndarray[np.double_t] pdf, np.ndarray[np.double_t] x):
    cdef int i
    cdef int n = len(pdf)
    cdef double lpdf
    cdef double rpdf
    cdef bw
    cdef np.ndarray[np.double_t] ret
    ret = np.zeros(n)
    ret[0] = 0
    for i in range(1,n):#do a trapezoid sum
        lpdf = pdf[i]
        rpdf = pdf[i-1]
        bw = x[i]-x[i-1]
        ret[i] = 0.5*(lpdf+rpdf)*bw + ret[i-1]
    return ret

#invert cdf useful for making toys
cpdef invert_cdf(np.ndarray[np.double_t] r,
                np.ndarray[np.double_t] cdf,
                np.ndarray[np.double_t] x):
    cdef np.ndarray[np.int_t] loc = np.searchsorted(cdf,r,'right')
    cdef int n = len(r)
    cdef np.ndarray[np.double_t] ret = np.zeros(n)
    cdef int i = 0
    cdef int ind
    cdef double ly
    cdef double lx
    cdef double ry
    cdef double rx
    cdef double minv
    for i in range(n):
        ind = loc[i]
        #print ind,i,len(loc),len(x)
        ly = cdf[ind-1]
        ry = cdf[ind]
        lx = x[ind-1]
        rx = x[ind]
        minv = (rx-lx)/(ry-ly)
        ret[i] = minv*(r[i]-ly)+lx
    return ret
