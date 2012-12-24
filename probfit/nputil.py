import numpy as np

def mid(x):
    return (x[:-1]+x[1:])/2

def minmax(x):
    return min(x),max(x)

#for converting numpy array to double
def float2double(a):
    if a is None or a.dtype == np.float64:
        return a
    else:
        return a.astype(np.float64)

def bool2int(b):
    if b:
        return 1
    else:
        return 0
