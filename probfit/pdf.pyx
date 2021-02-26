# cython: embedsignature=True, language_level=2

cimport cython

from libc.math cimport exp, pow, fabs, log, sqrt, sinh, tgamma, abs, fabs, cosh, atan2, asinh, erf
cdef double pi = 3.14159265358979323846264338327
import numpy as np
cimport numpy as np
from .funcutil import MinimalFuncCode

np.import_array()

cdef double badvalue = 1e-300
cdef double smallestdiv = 1e-10

cdef class Polynomial:
    """
    Polynomial.

    .. math::
        f(x; c_i) = \sum_{i < \\text{order}} c_i x^i

    User can supply order as integer in which case it uses (c_0....c_n+1)
    default or the list of coefficient name which the first one will be the
    lowest order and the last one will be the highest order. (order=1 is
    a linear function)

    """
    cdef int order
    cdef public object func_code
    cdef public object func_defaults
    def __init__(self,order,xname='x'):
        varnames = None
        argcount = 0
        if isinstance(order, int):
            if order<0 : raise ValueError('order must be >=0')
            self.order = order
            varnames = ['c_%d'%i for i in range(order+1)]
        else: #use supply list of coeffnames #to do check if it's list of string
            if len(order)<=0: raise ValueError('need at least one coefficient')
            self.order = len(order)-1 #yep -1 think about it
            varnames = order
        varnames.insert(0,xname) #insert x in front
        self.func_code = MinimalFuncCode(varnames)
        self.func_defaults = None

    def integrate(self, tuple bound, int nint_subdiv, *arg):
        cdef double a, b
        a, b = bound
        cdef double ret = 0.
        for i in range(self.order+1):
            ai1 = pow(a,i+1)
            bi1 = pow(b,i+1)
            ret += 1./(i+1) * arg[i] * (bi1-ai1)
        return ret

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


cdef class HistogramPdf:
    cdef np.ndarray hy
    cdef np.ndarray binedges
    cdef public object func_code
    cdef public object func_defaults
    def __init__(self, hy, binedges, xname='x'):
        """
        A histogram PDF. User supplies a template histogram with bin contents and bin
        edges. The histogram does not have to be normalized. The resulting PDF is normalized.
        """
        # Normalize, so the integral is unity
        yint= hy*(binedges[1:]-binedges[:-1])
        self.hy= hy.astype(float)/float(yint.sum())
        self.binedges= binedges
        if len(binedges)!= len(hy)+1:
            raise ValueError('binedges must be exactly one entry more than hy')
        # Only one variable. The PDF shape is fixed
        varnames= [xname]
        self.func_code = MinimalFuncCode(varnames)
        self.func_defaults = None

    def integrate(self, tuple bound, int nint_subdiv=0, arg=None):
        # nint_subdiv is irrelevant, ignored.
        # bound usually is smaller than the histogram's bound.
        # Find where they are:
        edges= np.copy(self.binedges)
        [ib0,ib1]= np.digitize([bound[0],bound[1]], edges)
        ib0= max(ib0,0)
        ib1= min(ib1, len(edges)-1)
        edges[ib0-1]= max(edges[ib0-1],bound[0])
        edges[ib1]= min(edges[ib1],bound[1])
        ilo= max(0,ib0-1)
        ihi= ib1+1 if edges[ib1-1]!=edges[ib1] else ib1
        return (self.hy[ilo:ihi-1]*np.diff(edges[ilo:ihi])).sum()

    def __call__(self, *arg):
        cdef double x = arg[0]
        [i]= np.digitize([x], self.binedges)
        if i >0 and i<=len(self.hy):
            return self.hy[i-1]
        else:
            return 0.0


cdef class _JohnsonSU:
    """
    Normalized JohnsonSU [1]_.

    .. math::
        f(x; \\mu, \\sigma, \\nu, \\tau) = \\frac{1}{\\lambda \\sqrt{2\\pi}}
        \\frac{1}{\\sqrt{1 + \\left( \\frac{x - \\xi}{\\lambda} \\right)}}
        e^{-\\frac{1}{2} \\left( -\\nu + \\frac{1}{\\tau}
        \\sinh^{-1} \\left( \\frac{x - \\xi}{\\lambda} \\right)\\right)^{2}}

    where

    .. math::
        \\lambda = \\sigma \\times \\left(\\frac{1}{2}( \\exp(\\tau^{2}) - 1)
        \\left(\\exp(\\tau^{2}) \\cosh\\left(-\\nu \\tau \\right) + 1\\right)
        \\right)^{-\\frac{1}{2}} \\\\

    and

    .. math::
        \\xi = \\mu + \\lambda \\exp\\left(\\frac{\\tau^{2}}{2}\\right)\\sinh
        \\left( \\nu \\tau \\right)

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution

    """

    cdef public object func_code
    cdef public object func_defaults

    def __init__(self, xname='x'):

        varnames = [xname, "mean", "sigma", "nu", "tau"]
        self.func_code = MinimalFuncCode(varnames)
        self.func_defaults = None

    def integrate(self, tuple bound, int nint_subdiv=0, *arg):

        cdef double a, b
        a, b = bound

        cdef double mean = arg[0]
        cdef double sigma = arg[1]
        cdef double nu = arg[2]
        cdef double tau = arg[3]

        cdef double w = exp(tau * tau)
        cdef double omega = - nu * tau
        cdef double c = 0.5 * (w-1) * (w * cosh(2 * omega) + 1)
        c = pow(c, -0.5)

        cdef double _lambda = sigma * c
        cdef double xi = mean + _lambda * sqrt(w) * sinh(omega)
        cdef double zmax = (b - xi) / _lambda
        cdef double zmin = (a - xi) / _lambda
        cdef double rmax = -nu + asinh(zmax) / tau
        cdef double rmin = -nu + asinh(zmin) / tau

        cdef double ret = 0.

        ret = 0.5 * (erf(rmax / (sqrt(2))) - erf(rmin / (sqrt(2))))

        return ret

    def __call__(self, *arg):

        cdef double x = arg[0]
        cdef double mean = arg[1]
        cdef double sigma = arg[2]
        cdef double nu = arg[3]
        cdef double tau = arg[4]

        cdef double w = exp(tau * tau)
        cdef double omega = - nu * tau

        cdef double c = 0.5 * (w-1) * (w * cosh(2 * omega) + 1)
        c = pow(c, -0.5)

        cdef double _lambda = sigma * c
        cdef double xi = mean + _lambda * sqrt(w) * sinh(omega)
        cdef double z = (x - xi) / _lambda
        cdef double r = -nu + asinh(z) / tau

        cdef double ret = 1. / (tau * _lambda * sqrt(2 * pi))
        ret *= 1. / sqrt(z * z + 1)
        ret *= exp(-0.5 * r * r)

        return ret

johnsonSU = _JohnsonSU()

cpdef double doublegaussian(double x, double mean, double sigma_L, double sigma_R):
    """
    Unnormalized double gaussian

    .. math::
        f(x;mean,\sigma_L,\sigma_R) =
        \\begin{cases}
            \exp \left[ -\\frac{1}{2} \left(\\frac{x-mean}{\sigma_L}\\right)^2 \\right], & \mbox{if } x < mean \\\\
            \exp \left[ -\\frac{1}{2} \left(\\frac{x-mean}{\sigma_R}\\right)^2 \\right], & \mbox{if } x >= mean
        \end{cases}

    """
    cdef double ret = 0.
    cdef double sigma = 0.
    sigma = sigma_L if x < mean else sigma_R
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


cpdef double crystalball(double x, double alpha, double n, double mean, double sigma):
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

cpdef double doublecrystalball(double x, double alpha, double alpha2, double n, double n2, double mean, double sigma):
    """
    Unnormalized double crystal ball function
    A gaussian core with two power tails



    """
    cdef double d = 0.
    cdef double ret = 0
    cdef double A = 0
    cdef double B = 0
    cdef double A2 = 0
    cdef double B2 = 0
    if sigma < smallestdiv:
        ret = badvalue
    elif fabs(alpha) < smallestdiv:
        ret = badvalue
    elif n<1.:
        ret = badvalue
    else:
        d = (x-mean)/sigma
        if d<-alpha:
            al = fabs(alpha)
            A=pow(n/al,n)*exp(-al**2/2.)
            B=n/al-al
            ret = A*pow(B-d,-n)

        elif d < alpha2 :
            ret = exp(-0.5*d**2)
        else:
            al2 = fabs(alpha2)
            A2=pow(n2/al2,n2)*exp(-al2**2/2.)
            B2=n2/al2-al2
            ret = A2*pow(B2+d,-n2)
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


cpdef double cruijff(double x, double m_0, double sigma_L, double sigma_R, double alpha_L, double alpha_R):
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
    cdef double dm2 = (x-m_0)*(x-m_0)
    cdef double demon=0.
    cdef double ret=0.
    if x<m_0:
        denom = 2*sigma_L*sigma_L+alpha_L*dm2
        if denom<smallestdiv:
            return 0.
        return exp(-dm2/denom)
    else:
        denom = 2*sigma_R*sigma_R+alpha_R*dm2
        if denom<smallestdiv:
            return 0.
        return exp(-dm2/denom)


cdef class _Linear:
    """
    Linear function.

    .. math::
        f(x;m,c) = mx+c

    """
    cdef public object func_code
    cdef public object func_defaults
    @cython.embedsignature(True)
    def __init__(self):
        #unfortunately cython doesn't support docstring for builtin class
        #http://docs.cython.org/src/userguide/special_methods.html
        self.func_code = MinimalFuncCode(['x','m','c'])
        self.func_defaults = None

    def __call__(self, double x, double m, double c):
        cdef double ret = m*x+c
        return ret

    cpdef double integrate(self, tuple bound, int nint_subdiv, double m, double c):
        cdef double a, b
        a, b = bound
        return 0.5*m*(b**2-a**2)+c*(b-a)

linear = _Linear()


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
        if fabs(qa) < smallestdiv: return badvalue
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

    .. plot:: pyplots/pdf/cauchy_bw.py
        :class: lightbox

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

    .. plot:: pyplots/pdf/cauchy_bw.py
        :class: lightbox

    """
    cdef double xmg = (x-m)/gamma
    return 1/(pi*gamma*(1+xmg*xmg))


cdef class _Exponential:
    """
    Exponential [1]_.

    .. math::
        f(x;\\tau) =
        \\begin{cases}
            \\exp \\left(-\\lambda x \\right) & \\mbox{if } \\x \\geq 0 \\\\
            0 & \\mbox{if } \\x < 0
        \\end{cases}
        
    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Exponential_distribution

    """
    cdef public object func_code
    cdef public object func_defaults

    def __init__(self, xname='x'):

        varnames = [xname, "lambda"]
        self.func_code = MinimalFuncCode(varnames)
        self.func_defaults = None

    def __call__(self, *arg):

        cdef double x = arg[0]
        cdef double _lambda = arg[1]
        cdef double ret = 0

        if x >= 0:
            ret = _lambda * exp(x * -_lambda)
        else:
            ret = 0.

        return ret

    def cdf(self, x, _lambda):

        cdef double ret = 0

        if x >= 0:
            ret = 1. - exp(x * -_lambda)
        else:
            ret = 0.

        return ret

    def integrate(self, tuple bound, int nint_subdiv, *arg):

        cdef double a, b
        a, b = bound

        cdef double _lambda = arg[0]

        Fa = self.cdf(a, _lambda)
        Fb = self.cdf(b, _lambda)

        return Fb - Fa


exponential = _Exponential()
