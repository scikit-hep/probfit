#cython: embedsignature=True
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, fabs, log, tgamma, lgamma, log1p, sqrt
import plotting
from matplotlib import pyplot as plt
from _libstat cimport compute_nll, compute_chi2_f, compute_bin_chi2_f,\
                      csum, compute_bin_lh_f, integrate1d
from funcutil import FakeFuncCode, merge_func_code
from nputil import float2double, mid, minmax
from functor cimport construct_arg
from util import describe, remove_prefix

np.import_array()

cdef extern from "math.h":
    bint isnan(double x)

cdef class SimultaneousFit:

    cdef readonly list allf
    cdef readonly list allpos
    cdef readonly int numf
    cdef readonly object func_code
    cdef readonly object func_defaults
    cdef np.ndarray factors
    cdef readonly object prefix
    #FIXME: cache each part if called with same parameter
    def __init__(self, *arg, factors=None, prefix=None, skip_prefix=None):
        """
        __init__(self, *arg, factors=None, prefix=None):

        Construct Simultaneous fit from given cost functions.

        **Arguments**

            - **factors** Optional factor array. If not None, each cost function
              is scaled by `factors[i]` before being summed up. Default None.
            - **prefix** Optional list of prefix. Add prefix to variablename of
              each cost function so that you don't accidentally merge them.
              Default None.
            - **skip_prefix** list of variable names that prefix should not be
              applied to. Useful if you want to prefix them and shared some
              of them.
        """
        self.allf = list(arg)
        func_code, allpos = merge_func_code(*arg, prefix=prefix,
                                            skip_prefix=skip_prefix)
        self.numf = len(arg)
        self.func_code = func_code
        self.func_defaults = None
        self.allpos = allpos
        if factors is None:
            factors = np.array([1.]*len(arg))
        self.prefix = prefix
        self.factors = factors


    def __call__(self, *arg):
        cdef double ret = 0.
        cdef np.ndarray[np.int_t] pos
        cdef tuple thisarg
        for i in range(self.numf):
            pos = self.allpos[i]
            thisarg = construct_arg(arg, pos)
            ret += self.factors[i]*self.allf[i](*thisarg)
        return ret


    def args_and_error_for(self, findex, minuit=None, args=None, errors=None):
        """
        convert argument from minuit/dictionary/errors to argument and error
        for cost function with index **findex**
        """
        #dictionary lookup for minuit.values and minuit.errors
        ret_val = None
        ret_err = None
        i = findex
        if minuit is not None:
            keys = minuit.parameters
            p = self.prefix
            #values = dict((remove_prefix(k, p), v) for k,v in minuit.values.items())
            ret_val = construct_arg(minuit.args, self.allpos[i])
            errors = dict((remove_prefix(k, p), v) for k,v in minuit.errors.items())
            return ret_val, errors

        if isinstance(args, dict):
            parameters = describe(self)
            pos = self.allpos[i]
            ret_val = [ args[parameters[pos[j]]] for j in range(len(pos))]
        else:
            ret_val = construct_arg(args, self.allpos[i])

        if errors is not None:
            ret_err = errors

        return ret_val, ret_err


    def show(self, m=None):
        """
        Same thing as :meth:`draw`. But show the figure immediately.

        .. seealso::
            :meth:`draw` for arguments.

        """
        self.draw(m)
        plt.show()


    def draw(self, minuit=None, args=None, errors=None, **kwds):
        plotting.draw_simultaneous(self, minuit=minuit, args=args,
                                   errors=errors, **kwds)


cdef class UnbinnedLH:
    cdef readonly object f
    cdef readonly object weights
    cdef public object func_code
    cdef readonly np.ndarray data
    cdef readonly int data_len
    cdef readonly badvalue
    cdef readonly tuple last_arg
    cdef readonly bint extended
    cdef readonly tuple extended_bound
    cdef readonly int extended_nint
    def __init__(self, f, data , weights=None, extended=False,
                 extended_bound=None, extended_nint=1000, badvalue=-100000):
        """
        __init__(self, f, data , weights=None, badvalue=-100000)

        Construct -log(unbinned likelihood) from callable *f*
        and data points *data*. Currently can only do 1D fit.

        .. math::
            \\textrm{UnbinnedLH} = \sum_{x \in \\textrm{data}} - \log f(x, arg \ldots)
        **Arguments**

            - **f** callable object. PDF that describe the data. The parameters
              are parsed using iminuit's ``describe``. The first positional
              arguement is assumed to be independent parameter. For example:

              ::

                    def gauss(x, mu, sigma):#good
                        pass
                    def bad_gauss(mu, sigma, x):#bad
                        pass

            - **data** 1D array of data.
            - **weights** Optional 1D array of weights. Default None(all 1).
            - **badvalue** Optional number. The value that will be used to
              represent log(lh) (notice no minus sign). When the likelihood
              is <= 0. This usually indicate your PDF is faraway from minimum
              or your PDF parameter has gone into unphysical region and return
              negative probability density. This should be a large negative
              number so that iminuit will avoid those points. Default -100000.
            - **extended** Set to True for extended fit. Default False. If Set
              to True the following term is added to resulting negative log
              likelihood

              .. math::
                \\textrm{ext_term} = \\int_{x \in \\textrm{extended_bound}}f(x, args, \\ldots) \\textrm{d} x

            - **extended_bound** Bound for calculating extended term.
              Default None(minimum and maximum of data will be used).
            - **extended_nint** number of trapezoid pieces to sum up as 
              integral for extende Term. Default 1000.
        .. note::
            There is a notable lack of **sum_w2** for unbinned likelihood. I
            feel like the solutions are quite sketchy. There are multiple ways
            to implement it but they don't really scale correctly. If you
            feel like there is a correct way to implement it feel free to do so
            and write document telling people about the caveat.
        """
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.weights = weights
        #only make copy when type mismatch
        self.data = float2double(data)
        self.data_len = len(data)
        self.badvalue = badvalue
        self.extended = extended
        self.extended_bound = extended_bound
        self.extended_nint = extended_nint
        if extended and extended_bound is None:
            self.extended_bound = minmax(data)

    def __call__(self,*arg):
        """
        Compute sum of -log(lh) given positional arguments.
        Position argument will be passed to pdf with independent vairable
        from `data` is given as the frist argument.
        """
        cdef double nll = 0.
        cdef double extended_term = 0.
        self.last_arg = arg
        nll = compute_nll(self.f, self.data, self.weights, arg, self.badvalue)
        if self.extended:
            extended_term = integrate1d(self.f, self.extended_bound,
                                        self.extended_nint, arg)
            nll += extended_term
        return nll

    def draw(self, minuit=None, bins=100, ax=None, bound=None,
            parmloc=(0.05,0.95), nfbins=200, print_par=True, args=None,
            errors=None, parts=False):
        """
        Draw comparison between histogram of data and pdf.

        **Arguments**

            - **minuit** Optional but recommended ``iminuit.Minuit`` object.
              If minuit is not ``None``, the pdf will be drawn using minimum
              value from minuit and parameters and error will be shown.
              If minuit is ``None``, then pdf will be drawn using argument from
              the last call to ``__call__``. Default ``None``

            - **bins** number of bins for histogram. Default 100.

            - **ax** matplotlib axes. If not given it will be drawn on current
              axes ``gca()``.

            - **bound** bound for histogram. If ``None`` is given the bound
              will be automatically determined from the data.
              If you given PDF that's normalied to a region but some data is
              not within the bound the picture may look funny.

            - **parmloc** location of parameter print out. This is passed
              directy to legend loc named parameter. Default (0.05,0.95).

            - **nfbins** how many point pdf should be evaluated. Default 200.

            - **print_par** print parameters and error on the plot. Default
              True.

            - **args** Optional. If minuit is not given, parameter value is
              determined from args. This can be dictionary of the form
              `{'a':1.0, 'b':1.0}` or list of values. Default None.

            - **errors** Optional dictionary of errors. If minuit is not given,
              parameter errors are determined from **errors**. Default None.

        """
        return plotting.draw_ulh(self, minuit=minuit, bins=bins, ax=ax,
            bound=bound, parmloc=parmloc, nfbins=nfbins, print_par=print_par,
            args=args, errors=errors, parts=parts)


    def default_errordef(self):
        return 0.5


    def show(self,*arg,**kwd):
        """
        Same thing as :meth:`draw`. But show the figure immediately.

        .. seealso::
            :meth:`draw` for arguments.

        """
        self.draw(*arg,**kwd)
        plt.show()


cdef class BinnedLH:
    cdef readonly object f
    cdef readonly object vf
    cdef readonly object func_code
    cdef readonly np.ndarray h
    cdef readonly np.ndarray w
    cdef readonly np.ndarray w2
    cdef readonly double N
    cdef readonly np.ndarray edges
    cdef readonly np.ndarray midpoints
    cdef readonly np.ndarray binwidth
    cdef readonly int bins
    cdef readonly double mymin
    cdef readonly double mymax
    cdef readonly double badvalue
    cdef readonly tuple last_arg
    cdef readonly int ndof
    cdef readonly bint extended
    cdef readonly bint use_w2
    def __init__(self, f, data, bins=40, weights=None, bound=None,
            badvalue=1000000, extended=False, use_w2=False):
        """
        __init__(self, f, data, bins=40, weights=None, bound=None,
            badvalue=1000000, extended=False, use_w2=False)
        Create a Poisson Binned Likelihood object from given PDF **f** and
        **data** (raw points not histogram). Constant term and expected minimum
        are subtracted off (aka. log likelihood ratio). The exact calculation
        will depend on **extended** and **use_w2** keyword parameters.

        .. math::
            \\textrm{BinnedLH} = -\sum_{i \in bins} s_i \\times  \left(  h_i \\times \log (\\frac{E_i}{h_i}) + (h_i-E_i) \\right)

        where
            - :math:`h_i` is sum of weight of data in ith bin.
            - :math:`b_i` is the width of ith bin.
            - :math:`N` is total number of data. :math:`N = \sum_i h_i`.
            - :math:`E_i` is expected number of occupancy in ith bin from PDF
              calculated using average of pdf value at both sides of the bin
              :math:`l_i, r_i`. The definition for :math:`E_i` depends whether
              extended likelihood is requested.

              If extended likelihood is requested (``extended=True``):

              .. math::
                    E_i = \\frac{f(l_i, arg\ldots )+f(r_i, arg \ldots )}{2} \\times b_i

              If extended likelihood is NOT requested (``extended=False``):

              .. math::
                    E_i = \\frac{f(l_i, arg \ldots )+f(r_i, arg \ldots )}{2} \\times b_i \\times N

              .. note::
                    You are welcome to patch this with a the using real area.
                    So that, it's less sensitive to bin size. Last time I check
                    ROOFIT used **f** evaluated at midpoint.

            - :math:`s_i` is a scaled factor. It's 1 if ``sum_w2=False``.
              It's :math:`s_i = \\frac{h_i}{\sum_{j \in \\textrm{bin_i}} w_j^2}`
              if ``sum_w2=True``. The factor will scale the statistics to the
              unweighted data.

            .. note::
                You may wonder why there is :math:`h_i-E_i` added at the end
                for each term of the sum. They sum up to zero anyway.
                The reason is the precision near the minimum. If we taylor
                expand the logarithmic term near :math:`h_i\\approx E_i` then
                the first order term will be :math:`h_i-E_i`. Subtracting this
                term at the end gets us the nice pure parabolic behavior for
                each term at the minimum.

        **Arguments**

            - **f** callable object. PDF that describe the data. The parameters
              are parsed using iminuit's ``describe``. The first positional
              arguement is assumed to be independent parameter. For example:

              ::

                    def gauss(x, mu, sigma):#good
                        pass
                    def bad_gauss(mu, sigma, x):#bad
                        pass

            - **data** 1D array of data. This is raw data not histogrammed
              data.

            - **bins** number of bins data should be histogrammed. Default 40.

            - **weights** Optional 1D array of weights. Default ``None``
              (all 1's).

            - **bound** tuple(min,max). Histogram bound. If ``None`` is given,
              bound is automatically determined from data. Default None.

            - **badvalue** Optional number. The value that will be used to
              represent log(lh) (notice no minus sign). When the likelihood
              is <= 0. This usually indicate your PDF is faraway from minimum
              or your PDF parameter has gone into unphysical region and return
              negative probability density. This should be a large POSITIVE
              number so that iminuit will avoid those points. Default 100000.

            - **extended** Boolean whether this likelihood should be extended
              likelihood or not. Default False.

            - **use_w2** Scale -log likelihood so that to the original
              unweighted statistics. Default False.

        """
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.use_w2 = use_w2
        self.extended = extended

        if bound is None: bound = minmax(data)

        self.mymin, self.mymax = bound

        h,self.edges = np.histogram(data, bins, range=bound, weights=weights)

        self.h = float2double(h)
        self.N = csum(self.h)

        if weights is not None:
            self.w2,_ = np.histogram(data, bins, range=bound,
                                     weights=weights*weights)
        else:
            self.w2,_ = np.histogram(data, bins, range=bound, weights=None)

        self.w2 = float2double(self.w2)
        self.midpoints = mid(self.edges)
        self.binwidth = np.diff(self.edges)

        self.bins = bins
        self.badvalue = badvalue
        self.ndof = self.bins-(self.func_code.co_argcount-1)


    def __call__(self,*arg):
        """
        Calculate sum -log(poisson binned likelihood) given positional
        arguments
        """
        self.last_arg = arg
        ret = compute_bin_lh_f(self.f,
                                self.edges,
                                self.h, #histogram,
                                self.w2,
                                self.N, #sum of h
                                arg, self.badvalue,
                                self.extended, self.use_w2)
        return ret


    def draw(self, minuit=None, ax = None,
            parmloc=(0.05,0.95), nfbins=200, print_par=True,
            args=None, errors=None, parts=False):
        """
        Draw comparison between histogram of data and pdf.

        **Arguments**

            - **minuit** Optional but recommended ``iminuit.Minuit`` object.
              If minuit is not ``None``, the pdf will be drawn using minimum
              value from minuit and parameters and error will be shown.
              If minuit is ``None``, then pdf will be drawn using argument from
              the last call to ``__call__``. Default ``None``

            - **ax** matplotlib axes. If not given it will be drawn on current
              axes ``gca()``.

            - **parmloc** location of parameter print out. This is passed
              directy to legend loc named parameter. Default (0.05,0.95).

            - **nfbins** how many point pdf should be evaluated. Default 200.

            - **print_par** print parameters and error on the plot.
              Default True.
        """
        return plotting.draw_blh(self, minuit=minuit,
            ax=ax, parmloc=parmloc, nfbins=nfbins, print_par=print_par,
            args=args, errors=errors, parts=parts)


    def default_errordef(self):
        return 0.5


    def show(self,*arg,**kwd):
        """
        Same thing as :meth:`draw`. But show the figure immediately.

        .. seealso::
            :meth:`draw` for arguments.

        """
        self.draw(*arg,**kwd)
        plt.show()


#fit a line with given function using minimizing chi2
cdef class Chi2Regression:
    cdef readonly object f
    cdef readonly object weights
    cdef readonly object error
    cdef readonly object func_code
    cdef readonly int data_len
    cdef readonly double badvalue
    cdef readonly int ndof
    cdef readonly np.ndarray x
    cdef readonly np.ndarray y
    cdef readonly tuple last_arg


    def __init__(self, f, x, y, error=None, weights=None):
        """
        __init__(self, f, x, y, error=None, weights=None):

        Create :math:`\chi^2` regression object. This is for fitting funciton
        to data points(x,y) rather than fitting PDF to a distribution.

        .. math::
            \\textrm{Chi2Regression} = \sum_{i} w_i \\times \left( \\frac{f(x_i, arg \ldots) - y_i}{error_i} \\right)^2

        **Arguments**
            - **f** callable object to describe line given by (**x** , **y**).
              The first positional arugment of **f** is assumed to be
              independent variable. Ex:::

                    def gauss(x, mu, sigma):#good
                        pass
                    def bad_gauss(mu, sigma, x):#bad
                        pass

            - **x** 1D array of independent variable
            - **y** 1D array of expected **y**
            - **error** optional 1D error array. If ``None`` is given, it's
              assumed to be all 1's.
            - **weight** 1D array weight for each data point.
        """
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.weights = float2double(weights)
        self.error = float2double(error)
        self.x = float2double(x)
        self.y = float2double(y)
        self.data_len = len(x)
        self.ndof = self.data_len-1-len(describe(self))


    def __call__(self,*arg):
        """
        Compute :math:`\chi^2`
        """
        self.last_arg = arg
        return compute_chi2_f(self.f, self.x, self.y, self.error,
                              self.weights, arg)


    def default_errordef(self):
        return 1.0


    def draw(self, minuit=None, ax=None, parmloc=(0.05,0.95), print_par=True,
             args=None, errors=None):
        """
        Draw comparison between points (**x**,**y**) and the function **f**.

        **Arguments**

            - **minuit** Optional but recommended ``iminuit.Minuit`` object.
              If minuit is not ``None``, the pdf will be drawn using minimum
              value from minuit and parameters and error will be shown.
              If minuit is ``None``, then pdf will be drawn using argument from
              the last call to ``__call__``. Default ``None``

            - **ax** matplotlib axes. If not given it will be drawn on current
              axes ``gca()``.

            - **parmloc** location of parameter print out. This is passed
              directy to legend loc named parameter. Default (0.05,0.95).

            - **print_par** print parameters and error on the plot.
              Default True.
        """
        return plotting.draw_x2(self, minuit=minuit, ax=ax, parmloc=parmloc,
                print_par=print_par, args=args, errors=errors)


    def show(self,*arg, **kwd):
        """
        Same thing as :meth:`draw`. But show the figure immediately.

        .. seealso::
            :meth:`draw` for arguments.

        """
        self.draw(*arg, **kwd)
        plt.show()


cdef class BinnedChi2:
    cdef readonly object f
    cdef readonly object vf
    cdef readonly object func_code
    cdef readonly np.ndarray h
    cdef readonly np.ndarray err
    cdef readonly np.ndarray edges
    cdef readonly np.ndarray midpoints
    cdef readonly np.ndarray binwidth
    cdef readonly int bins
    cdef readonly double mymin
    cdef readonly double mymax
    cdef readonly tuple last_arg
    cdef readonly int ndof
    def __init__(self, f, data, bins=40, weights=None, bound=None,
                 sumw2=False):
        """
        __init__(self, f, data, bins=40, weights=None, bound=None,
                 sumw2=False):

        Create Binned Chi2 Object. It calculates chi^2 assuming poisson
        statistics.

        .. math::
            \\textrm{BinnedChi2} = \sum_{i \in \\textrm{bins}} \left( \\frac{h_i - f(x_i,arg \ldots) \\times b_i \\times N}{err_i} \\right)^2

        Where :math:`err_i` is

            - :math:`\sqrt{\sum_{j \in \\textrm{bin}_i} w_j}` if ``sum_w2=False``.
            - :math:`\sqrt{\sum_{j \in \\textrm{bin}_i} w_j^2}` if ``sum_w2=True``.

        **Arguments**

            - **f** callable object. PDF describing **data**. The first
              positional arugment of **f** is assumed to be independent
              variable. Ex:::

                    def gauss(x, mu, sigma):#good
                        pass
                    def bad_gauss(mu, sigma, x):#bad
                        pass

            - **data** 1D array data (raw not histogram)

            - **bins** Optional number of bins to histogram data. Default 40.

            - **weights** 1D array weights.

            - **bound** tuple(min,max) bound of histogram. If ``None`` is given
              it's automatically determined from the data.

            - **sumw2** scale the error using
              :math:`\sqrt{\sum_{j \in \\textrm{bin}_i} w_j^2}`.
        """
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        if bound is None:
            bound = minmax(data)
        self.mymin,self.mymax = bound

        h,self.edges = np.histogram(data,bins,range=bound,weights=weights)

        self.h = float2double(h)
        self.midpoints = mid(self.edges)
        self.binwidth = np.diff(self.edges)

        #sumw2 if requested
        if weights is not None and sumw2:
            w2 = weights*weights
            sw2,_ = np.histogram(data,bins,range=bound,weights=w2)
            self.err = np.sqrt(sw2)
        else:
            self.err = np.sqrt(self.h)

        #check if error is too small
        if np.any(self.err<1e-5):
            raise ValueError('some bins are too small to do a chi2 fit. change your range')

        self.bins = bins
        self.ndof = self.bins-1-len(describe(self)) # fix this taking care of fixed parameter


    #lazy mid point implementation
    def __call__(self,*arg):
        """
        Calculate :math:`\chi^2` given positional arguments
        """
        self.last_arg = arg
        return compute_bin_chi2_f(self.f, self.midpoints, self.h, self.err,
                                  self.binwidth, None, arg)


    def draw(self, minuit=None, ax = None, parmloc=(0.05,0.95), nfbins=200,
             print_par=True, args=None, errors=None, parts=False):
        """
        Draw comparison histogram of data and the function **f**.

        **Arguments**

            - **minuit** Optional but recommended ``iminuit.Minuit`` object.
              If minuit is not ``None``, the pdf will be drawn using minimum
              value from minuit and parameters and error will be shown.
              If minuit is ``None``, then pdf will be drawn using argument from
              the last call to ``__call__``. Default ``None``

            - **ax** matplotlib axes. If not given it will be drawn on current
              axes ``gca()``.

            - **parmloc** location of parameter print out. This is passed
              directy to legend loc named parameter. Default (0.05,0.95).

            - **nfbins** number of points to calculate f

            - **print_par** print parameters and error on the plot.
              Default True.
        """
        return plotting.draw_bx2(self, minuit=minuit, ax=ax,
            parmloc=parmloc, nfbins=nfbins, print_par=print_par,
            args=args, errors=errors, parts=parts)


    def default_errordef(self):
        return 1.0


    def show(self,*arg,**kwd):
        """
        Same thing as :meth:`draw`. But show the figure immediately.

        .. seealso::
            :meth:`draw` for arguments.

        """
        self.draw(*arg,**kwd)
        plt.show()
