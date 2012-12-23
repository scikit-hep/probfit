import numpy as np
cimport numpy as np

cpdef np.ndarray compute_cdf(np.ndarray[np.double_t] pdf, np.ndarray[np.double_t] x):
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
