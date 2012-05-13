cimport numpy as np
cdef double cintegrate1d(f, tuple bound, int nint, tuple arg=*)
cdef double csum(np.ndarray x)
cdef np.ndarray[np.double_t] cvectorize_f(f,np.ndarray[np.double_t] x,tuple arg)

cdef double cxlogyx(double x,double y)
cdef double cwlogyx(double w,double y, double x)
