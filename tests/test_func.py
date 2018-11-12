from math import log
import numpy as np
from numpy.testing import assert_allclose
from iminuit import describe
from probfit import pdf
from probfit._libstat import xlogyx, wlogyx, csum, integrate1d, _vector_apply
from probfit.functor import construct_arg, fast_tuple_equal
from probfit.funcutil import merge_func_code


def f(x, y, z):
    return x + y + z


def f2(x, z, a):
    return x + z + a


def g(x, a, b):
    return x + a + b


def h(x, c, d):
    return x + c + d


def k_1(y, z):
    return y + z


def k_2(i, j):
    return i + j


# cpdef double doublegaussian(double x, double mean,
#                            double sigma_L, double sigma_R)
def test_doublegaussian():
    assert describe(pdf.doublegaussian) == ['x', 'mean', 'sigma_L', 'sigma_R']
    assert_allclose(pdf.doublegaussian(0., 0., 1., 2.), 1.)
    assert_allclose(pdf.doublegaussian(-1., 0., 1., 2.), 0.6065306597126334)
    assert_allclose(pdf.doublegaussian(1., 0., 1., 2.), 0.8824969025845955)


# cpdef double ugaussian(double x, double mean, double sigma)
def test_ugaussian():
    assert describe(pdf.ugaussian) == ['x', 'mean', 'sigma']
    assert_allclose(pdf.ugaussian(0, 0, 1), 1.)
    assert_allclose(pdf.ugaussian(-1, 0, 1), 0.6065306597126334)
    assert_allclose(pdf.ugaussian(1, 0, 1), 0.6065306597126334)


# cpdef double gaussian(double x, double mean, double sigma)
def test_gaussian():
    assert describe(pdf.gaussian) == ['x', 'mean', 'sigma']
    assert_allclose(pdf.gaussian(0, 0, 1), 0.3989422804014327)
    assert_allclose(pdf.gaussian(-1, 0, 1), 0.24197072451914337)
    assert_allclose(pdf.gaussian(1, 0, 1), 0.24197072451914337)


# cpdef double crystalball(double x,double alpha,double n,double mean,double sigma)
def test_crystalball():
    assert describe(pdf.crystalball) == ['x', 'alpha', 'n', 'mean', 'sigma']
    assert_allclose(pdf.crystalball(10, 1, 2, 10, 2), 1.)
    assert_allclose(pdf.crystalball(11, 1, 2, 10, 2), 0.8824969025845955)
    assert_allclose(pdf.crystalball(12, 1, 2, 10, 2), 0.6065306597126334)
    assert_allclose(pdf.crystalball(14, 1, 2, 10, 2), 0.1353352832366127)
    assert_allclose(pdf.crystalball(6, 1, 2, 10, 2), 0.26956918209450376)

# cpdef double doubecrystalball(double x,double alpha,double alpha2, double n,double n2, double mean,double sigma)
def test_doublecrystalball():
    assert describe(pdf.doublecrystalball) == ['x', 'alpha', 'alpha2', 'n', 'n2', 'mean', 'sigma']
    assert_allclose(pdf.doublecrystalball(10, 1, 1, 2, 2, 10, 2), 1.)
    assert_allclose(pdf.doublecrystalball(11, 1, 1, 2, 2, 10, 2), 0.8824969025845955)
    assert_allclose(pdf.doublecrystalball(12, 1, 1, 2, 2, 10, 2), 0.6065306597126334)
    assert_allclose(pdf.doublecrystalball(14, 1, 1, 2, 2, 10, 2), 0.26956918209450376)
    assert_allclose(pdf.doublecrystalball(6, 1, 1, 2, 2, 10, 2), 0.26956918209450376)
    assert_allclose(pdf.doublecrystalball(-10, 1, 5, 3, 4, 10, 2), 0.00947704155801)
    assert_allclose(pdf.doublecrystalball(0, 1, 5, 3, 4, 10, 2), 0.047744395954055)
    assert_allclose(pdf.doublecrystalball(11, 1, 5, 3, 4, 10, 2), 0.8824969025846)
    assert_allclose(pdf.doublecrystalball(20, 1, 5, 3, 4, 10, 2), 0.0000037266531720786)
    assert_allclose(pdf.doublecrystalball(25, 1, 5, 3, 4, 10, 2), 0.00000001287132228271)


    
# cpdef double argus(double x, double c, double chi, double p)
def test_argus():
    assert describe(pdf.argus) == ['x', 'c', 'chi', 'p']
    assert_allclose(pdf.argus(6., 10, 2, 3), 0.004373148605400128)
    assert_allclose(pdf.argus(10., 10, 2, 3), 0.)
    assert_allclose(pdf.argus(8., 10, 2, 3), 0.0018167930603254737)


# cpdef double cruijff(double x, double m_0, double sigma_L, double sigma_R, double alpha_L, double alpha_R)
def test_cruijff():
    assert describe(pdf.cruijff) == ['x', 'm_0', 'sigma_L', 'sigma_R', 'alpha_L', 'alpha_R']
    val = pdf.cruijff(0, 0, 1., 2., 1., 2.)
    assert_allclose(val, 1.)
    vl = pdf.cruijff(0, 1, 1., 1., 2., 2.)
    vr = pdf.cruijff(2, 1, 1., 1., 2., 2.)
    assert_allclose(vl, vr)
    assert_allclose(vl, 0.7788007830714)
    assert_allclose(vr, 0.7788007830714)


# cpdef double linear(double x, double m, double c)
def test_linear():
    assert describe(pdf.linear) == ['x', 'm', 'c']
    assert_allclose(pdf.linear(1, 2, 3), 5)
    assert hasattr(pdf.linear, 'integrate')
    integral = pdf.linear.integrate((0., 1.), 1, 1, 1)
    assert_allclose(integral, 1.5)


# cpdef double poly2(double x, double a, double b, double c)
def test_poly2():
    assert describe(pdf.poly2) == ['x', 'a', 'b', 'c']
    assert_allclose(pdf.poly2(2, 3, 4, 5), 25)


# cpdef double poly3(double x, double a, double b, double c, double d)
def test_poly3():
    assert describe(pdf.poly3) == ['x', 'a', 'b', 'c', 'd']
    assert_allclose(pdf.poly3(2, 3, 4, 5, 6), 56.)


def test_polynomial():
    p = pdf.Polynomial(1)
    assert describe(p) == ['x', 'c_0', 'c_1']
    assert_allclose(p(2, 2, 1), 4)
    integral = p.integrate((0, 1), 1, 2, 1)
    assert_allclose(integral, 2.5)

    p = pdf.Polynomial(2)
    assert describe(p) == ['x', 'c_0', 'c_1', 'c_2']
    assert_allclose(p(2, 3, 4, 5), 31)
    integral = p.integrate((2, 10), 10, 1, 2, 3)
    analytical = 8 + 2 / 2. * (10 ** 2 - 2 ** 2) + 3 / 3. * (10 ** 3 - 2 ** 3)
    assert_allclose(integral, analytical)


# cpdef double novosibirsk(double x, double width, double peak, double tail)
def test_novosibirsk():
    assert describe(pdf.novosibirsk) == ['x', 'width', 'peak', 'tail']
    assert_allclose(pdf.novosibirsk(3, 2, 3, 4), 1.1253517471925912e-07)


def test_rtv_breitwigner():
    assert describe(pdf.rtv_breitwigner) == ['x', 'm', 'gamma']
    assert_allclose(pdf.rtv_breitwigner(1, 1, 1.), 0.8194496535636714)
    assert_allclose(pdf.rtv_breitwigner(1, 1, 2.), 0.5595531041435416)
    assert_allclose(pdf.rtv_breitwigner(1, 2, 3.), 0.2585302502852219)


def test_cauchy():
    assert describe(pdf.cauchy), ['x', 'm', 'gamma']
    assert_allclose(pdf.cauchy(1, 1, 1.), 0.3183098861837907)
    assert_allclose(pdf.cauchy(1, 1, 2.), 0.15915494309189535)
    assert_allclose(pdf.cauchy(1, 2, 4.), 0.07489644380795074)

def test_johnsonSU():
    assert describe(pdf.johnsonSU), ['x', "mean", "sigma", "nu", "tau"]
    assert_allclose(pdf.johnsonSU(1., 1., 1., 1., 1.), 0.5212726124342)
    assert_allclose(pdf.johnsonSU(1., 2., 1., 1., 1.), 0.1100533373219)
    assert_allclose(pdf.johnsonSU(1., 2., 2., 1., 1.), 0.4758433826682)

    j = pdf.johnsonSU
    assert (hasattr(j, 'integrate'))
    integral = j.integrate((-100, 100), 0, 1., 1., 1., 1.)
    assert_allclose(integral, 1.0)
    integral = j.integrate((0, 2), 0, 1., 1., 1., 1.)
    assert_allclose(integral, 0.8786191859)

def test_HistogramPdf():
    be = np.array([0, 1, 3, 4], dtype=float)
    hy = np.array([10, 30, 50], dtype=float)
    norm = float((hy * np.diff(be)).sum())
    f = pdf.HistogramPdf(hy, be)
    assert_allclose(f(0.5), 10.0 / norm)
    assert_allclose(f(1.2), 30.0 / norm)
    assert_allclose(f(2.9), 30.0 / norm)
    assert_allclose(f(3.6), 50.0 / norm)

    assert (hasattr(f, 'integrate'))

    integral = f.integrate((0, 4))
    assert_allclose(integral, 1.0)
    integral = f.integrate((0.5, 3.4))
    assert_allclose(integral, (10 * 0.5 + 30 * 2 + 50 * 0.4) / norm)
    integral = f.integrate((1.2, 4.5))
    assert_allclose(integral, (30 * 1.8 + 50 * 1) / norm)


def test__vector_apply():
    def f(x, y):
        return x * x + y

    y = 10
    a = np.array([1., 2., 3.])
    expected = [f(x, y) for x in a]
    va = _vector_apply(f, a, tuple([y]))
    assert_allclose(va, expected)


def test_integrate1d():
    def f(x, y):
        return x * x + y

    def intf(x, y):
        return x * x * x / 3. + y * x

    bound = (-2., 1.)
    y = 3.
    integral = integrate1d(f, bound, 1000, tuple([y]))
    analytic = intf(bound[1], y) - intf(bound[0], y)
    assert_allclose(integral, analytic)


def test_integrate1d_analytic():
    class temp:
        def __call__(self, x, m, c):
            return m * x ** 2 + c

        def integrate(self, bound, nint, m, c):
            a, b = bound
            return b - a  # (wrong on purpose)

    bound = (0., 10.)
    f = temp()
    integral = integrate1d(f, bound, 10, (2., 3.))
    assert_allclose(integral, bound[1] - bound[0])


def test_csum():
    x = np.array([1, 2, 3], dtype=np.double)
    s = csum(x)
    assert_allclose(s, 6.)


def test_xlogyx():
    def bad(x, y):
        return x * log(y / x)

    assert_allclose(xlogyx(1., 1.), bad(1., 1.))
    assert_allclose(xlogyx(1., 2.), bad(1., 2.))
    assert_allclose(xlogyx(1., 3.), bad(1., 3.))
    assert_allclose(xlogyx(0., 1.), 0.)


def test_wlogyx():
    def bad(w, y, x):
        return w * log(y / x)

    assert_allclose(wlogyx(1., 1., 1.), bad(1., 1., 1.))
    assert_allclose(wlogyx(1., 2., 3.), bad(1., 2., 3.))
    assert_allclose(wlogyx(1e-50, 1e-20, 1.), bad(1e-50, 1e-20, 1.))


def test_construct_arg():
    arg = (1, 2, 3, 4, 5, 6)
    pos = np.array([0, 2, 4], dtype=np.int)
    carg = construct_arg(arg, pos)
    assert carg == (1, 3, 5)


def test_merge_func_code():
    funccode, [pf, pg, ph] = merge_func_code(f, g, h)
    assert funccode.co_varnames == ('x', 'y', 'z', 'a', 'b', 'c', 'd')
    assert tuple(pf) == (0, 1, 2)
    assert tuple(pg) == (0, 3, 4)
    assert tuple(ph) == (0, 5, 6)


def test_merge_func_code_prefix():
    funccode, [pf, pg, ph] = merge_func_code(
        f, g, h,
        prefix=['f_', 'g_', 'h_'],
        skip_first=True)
    expected = 'x', 'f_y', 'f_z', 'g_a', 'g_b', 'h_c', 'h_d'
    assert funccode.co_varnames == expected
    assert tuple(pf) == (0, 1, 2)
    assert tuple(pg) == (0, 3, 4)
    assert tuple(ph) == (0, 5, 6)


def test_merge_func_code_factor_list():
    funccode, [pf, pg, pk_1, pk_2] = merge_func_code(
        f, g,
        prefix=['f_', 'g_'],
        skip_first=True,
        factor_list=[k_1, k_2])
    expected = 'x', 'f_y', 'f_z', 'g_a', 'g_b', 'g_i', 'g_j'
    assert funccode.co_varnames == expected

    assert tuple(pf) == (0, 1, 2)
    assert tuple(pg) == (0, 3, 4)
    assert tuple(pk_1) == (1, 2)
    assert tuple(pk_2) == (5, 6)


def test_merge_func_code_skip_prefix():
    funccode, _ = merge_func_code(
        f, f2,
        prefix=['f_', 'g_'],
        skip_first=True,
        skip_prefix=['z'])
    assert funccode.co_varnames == ('x', 'f_y', 'z', 'g_a')


def test_fast_tuple_equal():
    a = (1., 2., 3.)
    b = (1., 2., 3.)
    assert fast_tuple_equal(a, b, 0) is True

    a = (1., 4., 3.)
    b = (1., 2., 3.)
    assert fast_tuple_equal(a, b, 0) is False

    a = (4., 3.)
    b = (1., 4., 3.)
    assert fast_tuple_equal(a, b, 1) is True

    a = (4., 5.)
    b = (1., 4., 3.)
    assert fast_tuple_equal(a, b, 1) is False

    a = tuple([])
    b = tuple([])
    assert fast_tuple_equal(a, b, 0) is True
