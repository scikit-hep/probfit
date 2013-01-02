#cython: embedsignature=True
cimport cython

from libc.math cimport exp, pow, fabs, log, sqrt, sinh, tgamma, log1p, abs
cdef double pi = 3.14159265358979323846264338327
import numpy as np
cimport numpy as np
from util import describe
from funcutil import MinimalFuncCode
np.import_array()

cdef double badvalue = 1e-300
cdef double smallestdiv = 1e-10

cdef class Polynomial:
    cdef int order
    cdef public object func_code
    cdef public object func_defaults
    def __init__(self,order,xname='x'):
        """
        .. math::
            f(x; c_i) = \sum_{i < \\text{order}} c_i x^i

        User can supply order as integer in which case it uses (c_0....c_n+1)
        default or the list of coefficient name which the first one will be the
        lowest order and the last one will be the highest order
        """
        varnames = None
        argcount = 0
        if isinstance(order, int):
            if order<0 : raise ValueError('order must be >=0')
            self.order = order
            varnames = ['c_%d'%i for i in range(order+1)]
        else: #use supply list of coeffnames #to do check if it's list of string
            if len(order)<=0: raise ValueError('need at least one coefficient')
            self.order=len(order)-1 #yep -1 think about it
            varnames = order
        varnames.insert(0,xname) #insert x in front
        self.func_code = MinimalFuncCode(varnames)
        self.func_defaults = None

    def __call__(self,*arg):
        cdef double x = arg[0]
        cdef double t
        cdef double ret=0.
        cdef int iarg
        cdef int numarg = self.order+1 #number of coefficient
        cdef int i
        for i in range(numarg):
            iarg = i+1
            t = arg[iarg]
            if i > 0:
                ret+=pow(x,i)*t
            else:#avoid 0^0
                ret += t
        return ret


cpdef double doublegaussian(double x, double mean, double sigmal, double sigmar):
    """
    Unnormalized double gaussian

    .. math::
        f(x;mean,\sigma_l,\sigma_r) =
        \\begin{cases}
            \exp \left[ -\\frac{1}{2} \left(\\frac{x-mean}{\sigma_l}\\right)^2 \\right], & \mbox{if } x < mean \\\\
            \exp \left[ -\\frac{1}{2} \left( -\\frac{x-mean}{\sigma_r}\\right)^2 \\right], & \mbox{if } x >= mean
        \end{cases}

    """
    cdef double ret = 0.
    cdef double sigma = 0.
    sigma = sigmal if x < mean else sigmar
    if sigma < smallestdiv:
        ret = badvalue
    else:

        d = (x-mean)/sigma
        d2 = d*d
        ret = exp(-0.5*d2)
    return ret

cpdef double ugaussian(double x, double mean, double sigma):
    """
    Unnormalized gaussian

    .. math::
        f(x; mean, \sigma) = \exp \left[ -\\frac{1}{2} \\left( \\frac{x-mean}{\sigma} \\right)^2 \\right]

    """
    cdef double ret = 0
    if sigma < smallestdiv:
        ret = badvalue
    else:
        d = (x-mean)/sigma
        d2 = d*d
        ret = exp(-0.5*d2)
    return ret


cpdef double gaussian(double x, double mean, double sigma):
    """
    Normalized gaussian.

    .. math::
        f(x; mean, \sigma) = \\frac{1}{\sqrt{2\pi}\sigma}
        \exp \left[  -\\frac{1}{2} \left(\\frac{x-mean}{\sigma}\\right)^2 \\right]
    """
    cdef double badvalue = 1e-300
    cdef double ret = 0
    if sigma < smallestdiv:
        ret = badvalue
    else:
        d = (x-mean)/sigma
        d2 = d*d
        ret = 1/(sqrt(2*pi)*sigma)*exp(-0.5*d2)
    return ret


cpdef double crystalball(double x,double alpha,double n,double mean,double sigma):
    """
    Unnormalized crystal ball function

    .. math::
        f(x;\\alpha,n,mean,\sigma) =
        \\begin{cases}
            \exp\left( -\\frac{1}{2} \delta^2 \\right) & \mbox{if } \delta>-\\alpha \\\\
            \left( \\frac{n}{|\\alpha|} \\right)^n \left( \\frac{n}{|\\alpha|} - |\\alpha| - \delta \\right)^{-n} 
            \exp\left( -\\frac{1}{2}\\alpha^2\\right)
            & \mbox{if } \delta \leq \\alpha
        \end{cases}

    where
        - :math:`\delta = \\frac{x-mean}{\sigma}`

    .. note::
        http://en.wikipedia.org/wiki/Crystal_Ball_function
    """
    cdef double d = 0.
    cdef double ret = 0
    cdef double A = 0
    cdef double B = 0
    if sigma < smallestdiv:
        ret = badvalue
    elif fabs(alpha) < smallestdiv:
        ret = badvalue
    elif n<1.:
        ret = badvalue
    else:
        d = (x-mean)/sigma
        if d > -alpha :
            ret = exp(-0.5*d**2)
        else:
            al = fabs(alpha)
            A=pow(n/al,n)*exp(-al**2/2.)
            B=n/al-al
            ret = A*pow(B-d,-n)
    return ret


#Background stuff
cpdef double argus(double x, double c, double chi, double p):
    """
    Unnormalized argus distribution

    .. math::
        f(x;c,\chi,p) = \\frac{x}{c^2} \left( 1-\\frac{x^2}{c^2} \\right)
            \exp \left( - \\frac{1}{2} \chi^2 \left( 1 - \\frac{x^2}{c^2} \\right) \\right)

    .. note::
        http://en.wikipedia.org/wiki/ARGUS_distribution
    """
    if c<smallestdiv:
        return badvalue
    if x>c:
        return 0.
    cdef double xc = x/c
    cdef double xc2 = xc*xc
    cdef double ret = 0

    ret = xc/c*pow(1.-xc2,p)*exp(-0.5*chi*chi*(1-xc2))

    return ret


cpdef double cruijff(double x, double m0, double sigma_L, double sigma_R, double alpha_L, double alpha_R):
    """
    Unnormalized cruijff function

    .. math::
        f(x;m_0, \sigma_L, \sigma_R, \\alpha_L, \\alpha_R) =
        \\begin{cases}
            \exp\left(-\\frac{(x-m_0)^2}{2(\sigma_{L}^2+\\alpha_{L}(x-m_0)^2)}\\right)
            & \mbox{if } x<m_0 \\\\
            \exp\left(-\\frac{(x-m_0)^2}{2(\sigma_{R}^2+\\alpha_{R}(x-m_0)^2)}\\right)
            & \mbox{if } x<m_0 \\\\
        \end{cases}

    """
    cdef double dm2 = (x-m0)*(x-m0)
    cdef double demon=0.
    cdef double ret=0.
    if x<m0:
        denom = 2*sigma_L*sigma_L+alpha_L*dm2
        if denom<smallestdiv:
            return 0.
        return exp(-dm2/denom)
    else:
        denom = 2*sigma_R*sigma_R+alpha_R*dm2
        if denom<smallestdiv:
            return 0.
        return exp(-dm2/denom)


cpdef double linear(double x, double m, double c):
    """
    Linear function.

    .. math::
        f(x;m,c) = mx+c
    """
    cdef double ret = m*x+c
    return ret


cpdef double poly2(double x, double a, double b, double c):
    """
    Parabola

    .. math::
        f(x;a,b,c) = ax^2+bx+c
    """
    cdef double ret = a*x*x+b*x+c
    return ret


cpdef double poly3(double x, double a, double b, double c, double d):
    """
    Polynomial of third order

    .. math::
        f(x; a,b,c,d) =ax^3+bx^2+cx+d

    """
    cdef double x2 = x*x
    cdef double x3 = x2*x
    cdef double ret = a*x3+b*x2+c*x+d
    return ret


cpdef double novosibirsk(double x, double width, double peak, double tail):
    """
    Unnormalized Novosibirsk

    .. math::
        f(x;\sigma, x_0, \Lambda) = \exp\left[
            -\\frac{1}{2} \\frac{\left( \ln q_y \\right)^2 }{\Lambda^2} + \Lambda^2 \\right] \\\\
        q_y(x;\sigma,x_0,\Lambda) = 1 + \\frac{\Lambda(x-x_0)}{\sigma} \\times
        \\frac{\sinh \left( \Lambda \sqrt{\ln 4} \\right)}{\Lambda \sqrt{\ln 4}}

    where
        - width = :math:`\sigma`
        - peak = :math:`m_0`
        - tail = :math:`\Lambda`
    """
    #credit roofit implementation
    cdef double qa
    cdef double qb
    cdef double qc
    cdef double qx
    cdef double qy
    cdef double xpw
    cdef double lqyt
    if width < smallestdiv: return badvalue
    xpw = (x-peak)/width
    if fabs(tail) < 1e-7:
        qc = 0.5*xpw*xpw
    else:
        qa = tail*sqrt(log(4.))
        if qa < smallestdiv: return badvalue
        qb = sinh(qa)/qa
        qx = xpw*qb
        qy = 1.+tail*qx
        if qy > 1e-7:
            lqyt = log(qy)/tail
            qc =0.5*lqyt*lqyt + tail*tail
        else:
            qc=15.
    return exp(-qc)


cpdef double rtv_breitwigner(double x, double m, double gamma):
    """
    Normalized Relativistic Breit-Wigner

    .. math::
        f(x; m, \Gamma) = N\\times \\frac{1}{(x^2-m^2)^2+m^2\Gamma^2}

    where

    .. math::
        N = \\frac{2 \sqrt{2} m \Gamma  \gamma }{\pi \sqrt{m^2+\gamma}}

    and

    .. math::
        \gamma=\sqrt{m^2\left(m^2+\Gamma^2\\right)}


    .. seealso::
        :func:`cauchy`

    """
    cdef double mm = m*m
    cdef double xm = x*x-mm
    cdef double gg = gamma*gamma
    cdef double s = sqrt(mm*(mm+gg))
    cdef double N = (2*sqrt(2)/pi)*m*gamma*s/sqrt(mm+s)
    return N/(xm*xm+mm*gg)


cpdef double cauchy(double x, double m, double gamma):
    """
    Cauchy distribution aka non-relativistic Breit-Wigner

    .. math::
        f(x, m, \gamma) = \\frac{1}{\pi \gamma \left[ 1+\left( \\frac{x-m}{\gamma} \\right)^2\\right]}

    .. seealso::
        :func:`breitwigner`
    """
    cdef double xmg = (x-m)/gamma
    return 1/(pi*gamma*(1+xmg*xmg))
