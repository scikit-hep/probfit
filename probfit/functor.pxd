cimport numpy as np

cpdef tuple construct_arg(tuple arg, np.ndarray[np.int_t] fpos)

cpdef bint fast_tuple_equal(tuple t1, tuple t2 , int t2_offset) except *
