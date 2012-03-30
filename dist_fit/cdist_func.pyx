cimport cython
from libc.math cimport exp,pow,fabs,log,sqrt

cdef double pi=3.1415926535897932384626433832795028841971693993751058209749445923078164062

#peaking stuff
@cython.binding(True)
def gaussian(double x, double mean, double sigma):
    """
    gaussian normalized for -inf to +inf
    """
    cdef double ret = 0
    if sigma < 1e-7:
        ret = 1e-300
    else:
        d = (x-mean)/sigma
        d2 = d*d
        ret = 1/(sqrt(2*pi)*sigma)*exp(0.5*d2)
    return ret

@cython.binding(True)
def crystalball(double x,double alpha,double n,double mean,double sigma):
    """
    unnormalized crystal ball function 
    see http://en.wikipedia.org/wiki/Crystal_Ball_function
    """
    cdef double d = (x-mean)/sigma
    cdef double ret = 0
    cdef double A = 0
    cdef double B = 0
    
    if d > -alpha : 
        ret = exp(-0.5*d**2)
    else:
        al = fabs(alpha)
        A=pow(n/al,n)*exp(-al**2/2.)
        B=n/al-al
        ret = A*pow(B-d,-n)
    return ret

#Background stuff
@cython.binding(True)
def argus(double x, double c, double chi):
    """
    unnormalized argus distribution
    see: http://en.wikipedia.org/wiki/ARGUS_distribution
    """
    cdef double xc = x/c
    cdef double xc2 = xc*xc
    cdef double ret = 0
    ret = xc/c*sqrt(1-xc2)*exp(-0.5*chi*chi*(1-xc2))
    return ret

#Polynomials
@cython.binding(True)
def linear(double x, double m, double c):
    """
    y=mx+c
    """
    cdef double ret = m*x+c
    return ret

@cython.binding(True)
def poly2(double x, double a, double b, double c):
    """
    y=ax^2+bx+c
    """
    cdef double ret = a*x*x+b*x+c
    return ret
@cython.binding(True)
def poly3(double x, double a, double b, double c, double d):
    """
    y=ax^2+bx+c
    """
    cdef double x2 = x*x
    cdef double x3 = x2*x
    cdef double ret = a*x3+b*x2+c*x+d
    return ret

