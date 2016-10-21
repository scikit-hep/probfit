from math import log
import numpy as np
from nose.tools import assert_equal, assert_almost_equal
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

def iterable_equal(x, y):
    assert_equal(list(x), list(y))

# cpdef double doublegaussian(double x, double mean,
#                            double sigma_L, double sigma_R)
def test_doublegaussian():
    assert_equal(
        tuple(describe(pdf.doublegaussian)), ('x', 'mean', 'sigma_L', 'sigma_R'))
    assert_almost_equal(pdf.doublegaussian(0., 0., 1., 2.), 1.)
    assert_almost_equal(pdf.doublegaussian(-1., 0., 1., 2.), 0.6065306597126334)
    assert_almost_equal(pdf.doublegaussian(1., 0., 1., 2.), 0.8824969025845955)

# cpdef double ugaussian(double x, double mean, double sigma)
def test_ugaussian():
    assert_equal(tuple(describe(pdf.ugaussian)), ('x', 'mean', 'sigma'))
    assert_almost_equal(pdf.ugaussian(0, 0, 1), 1.)
    assert_almost_equal(pdf.ugaussian(-1, 0, 1), 0.6065306597126334)
    assert_almost_equal(pdf.ugaussian(1, 0, 1), 0.6065306597126334)

# cpdef double gaussian(double x, double mean, double sigma)
def test_gaussian():
    assert_equal(tuple(describe(pdf.gaussian)), ('x', 'mean', 'sigma'))
    assert_almost_equal(pdf.gaussian(0, 0, 1), 0.3989422804014327)
    assert_almost_equal(pdf.gaussian(-1, 0, 1), 0.24197072451914337)
    assert_almost_equal(pdf.gaussian(1, 0, 1), 0.24197072451914337)

# cpdef double crystalball(double x,double alpha,double n,double mean,double sigma)
def test_crystalball():
    assert_equal(tuple(describe(pdf.crystalball)),
                ('x', 'alpha', 'n', 'mean', 'sigma'))
    assert_almost_equal(pdf.crystalball(10, 1, 2, 10, 2), 1.)
    assert_almost_equal(pdf.crystalball(11, 1, 2, 10, 2), 0.8824969025845955)
    assert_almost_equal(pdf.crystalball(12, 1, 2, 10, 2), 0.6065306597126334)
    assert_almost_equal(pdf.crystalball(14, 1, 2, 10, 2), 0.1353352832366127)
    assert_almost_equal(pdf.crystalball(6, 1, 2, 10, 2), 0.26956918209450376)

# cpdef double argus(double x, double c, double chi, double p)
def test_argus():
    assert_equal(tuple(describe(pdf.argus)), ('x', 'c', 'chi', 'p'))
    assert_almost_equal(pdf.argus(6., 10, 2, 3), 0.004373148605400128)
    assert_almost_equal(pdf.argus(10., 10, 2, 3), 0.)
    assert_almost_equal(pdf.argus(8., 10, 2, 3), 0.0018167930603254737)

# cpdef double cruijff(double x, double m_0, double sigma_L, double sigma_R, double alpha_L, double alpha_R)
def test_cruijff():
    iterable_equal(tuple(describe(pdf.cruijff)),
        ('x', 'm_0', 'sigma_L', 'sigma_R', 'alpha_L', 'alpha_R'))
    val = pdf.cruijff(0, 0, 1., 2., 1., 2.)
    assert_almost_equal(val, 1.)
    vl = pdf.cruijff(0, 1, 1., 1., 2., 2.)
    vr = pdf.cruijff(2, 1, 1., 1., 2., 2.)
    assert_almost_equal(vl, vr, msg='symmetric test')
    assert_almost_equal(vl, 0.7788007830714)
    assert_almost_equal(vr, 0.7788007830714)

# cpdef double linear(double x, double m, double c)
def test_linear():
    assert_equal(describe(pdf.linear), ['x', 'm', 'c'])
    assert_almost_equal(pdf.linear(1, 2, 3), 5)
    assert(hasattr(pdf.linear, 'integrate'))
    integral = pdf.linear.integrate((0., 1.), 1, 1, 1)
    assert_equal(integral, 1.5)

# cpdef double poly2(double x, double a, double b, double c)
def test_poly2():
    assert_equal(describe(pdf.poly2), ['x', 'a', 'b', 'c'])
    assert_almost_equal(pdf.poly2(2, 3, 4, 5), 25)

# cpdef double poly3(double x, double a, double b, double c, double d)
def test_poly3():
    assert_equal(describe(pdf.poly3), ['x', 'a', 'b', 'c', 'd'])
    assert_almost_equal(pdf.poly3(2, 3, 4, 5, 6), 56.)

def test_polynomial():
    p = pdf.Polynomial(1)
    assert_equal(describe(p), ['x', 'c_0', 'c_1'])
    assert_equal(p(2, 2, 1), 4)
    integral = p.integrate((0, 1), 1, 2, 1)
    assert_equal(integral, 2.5)

    p = pdf.Polynomial(2)
    assert_equal(describe(p), ['x', 'c_0', 'c_1', 'c_2'])
    assert_equal(p(2, 3, 4, 5), 31)
    integral = p.integrate((2, 10), 10, 1, 2, 3)
    analytical = 8 + 2 / 2.*(10 ** 2 - 2 ** 2) + 3 / 3.*(10 ** 3 - 2 ** 3)    
    assert_equal(integral, analytical)


# cpdef double novosibirsk(double x, double width, double peak, double tail)
def test_novosibirsk():
    assert_equal(describe(pdf.novosibirsk), ['x', 'width', 'peak', 'tail'])
    assert_almost_equal(pdf.novosibirsk(3, 2, 3, 4), 1.1253517471925912e-07)


def test_rtv_breitwigner():
    assert_equal(describe(pdf.rtv_breitwigner), ['x', 'm', 'gamma'])
    assert_almost_equal(pdf.rtv_breitwigner(1, 1, 1.), 0.8194496535636714)
    assert_almost_equal(pdf.rtv_breitwigner(1, 1, 2.), 0.5595531041435416)
    assert_almost_equal(pdf.rtv_breitwigner(1, 2, 3.), 0.2585302502852219)


def test_cauchy():
    assert_equal(describe(pdf.cauchy), ['x', 'm', 'gamma'])
    assert_almost_equal(pdf.cauchy(1, 1, 1.), 0.3183098861837907)
    assert_almost_equal(pdf.cauchy(1, 1, 2.), 0.15915494309189535)
    assert_almost_equal(pdf.cauchy(1, 2, 4.), 0.07489644380795074)

def test_HistogramPdf():
    be = np.array([0, 1, 3, 4], dtype=float)
    hy = np.array([10, 30, 50], dtype=float)
    norm = float((hy * np.diff(be)).sum())
    f = pdf.HistogramPdf(hy, be)
    assert_almost_equal(f(0.5), 10.0 / norm)
    assert_almost_equal(f(1.2), 30.0 / norm)
    assert_almost_equal(f(2.9), 30.0 / norm)
    assert_almost_equal(f(3.6), 50.0 / norm)

    assert(hasattr(f, 'integrate'))

    integral = f.integrate((0, 4))
    assert_almost_equal(integral, 1.0)
    integral = f.integrate((0.5, 3.4))
    assert_almost_equal(integral, (10 * 0.5 + 30 * 2 + 50 * 0.4) / norm)
    integral = f.integrate((1.2, 4.5))
    assert_almost_equal(integral, (30 * 1.8 + 50 * 1) / norm)


def test__vector_apply():
    def f(x, y):
        return x * x + y
    y = 10
    a = np.array([1., 2., 3.])
    expected = [f(x, y) for x in a]
    va = _vector_apply(f, a, tuple([y]))
    for i in range(len(a)):
        assert_almost_equal(va[i], expected[i])

def test_integrate1d():
    def f(x, y):
        return x * x + y

    def intf(x, y):
        return x * x * x / 3. + y * x
    bound = (-2., 1.)
    y = 3.
    integral = integrate1d(f, bound, 1000, tuple([y]))
    analytic = intf(bound[1], y) - intf(bound[0], y)
    assert_almost_equal(integral, analytic)

def test_integrate1d_analytic():
    class temp:
        def __call__(self, x, m , c):
            return m * x ** 2 + c
        def integrate(self, bound, nint, m, c):
            a, b = bound
            return b - a  # (wrong on purpose)
    bound = (0., 10.)
    f = temp()
    integral = integrate1d(f, bound, 10, (2., 3.))
    assert_equal(integral, bound[1] - bound[0])

def test_csum():
    x = np.array([1, 2, 3], dtype=np.double)
    s = csum(x)
    assert_almost_equal(s, 6.)


def test_xlogyx():
    def bad(x, y):
        return x * log(y / x)
    assert_almost_equal(xlogyx(1., 1.), bad(1., 1.))
    assert_almost_equal(xlogyx(1., 2.), bad(1., 2.))
    assert_almost_equal(xlogyx(1., 3.), bad(1., 3.))
    assert_almost_equal(xlogyx(0., 1.), 0.)


def test_wlogyx():
    def bad(w, y, x):
        return w * log(y / x)
    assert_almost_equal(wlogyx(1., 1., 1.), bad(1., 1., 1.))
    assert_almost_equal(wlogyx(1., 2., 3.), bad(1., 2., 3.))
    assert_almost_equal(wlogyx(1e-50, 1e-20, 1.), bad(1e-50, 1e-20, 1.))


def test_construct_arg():
    arg = (1, 2, 3, 4, 5, 6)
    pos = np.array([0, 2, 4], dtype=np.int)
    carg = construct_arg(arg, pos)
    expected = (1, 3, 5)
    # print carg
    for i in range(len(carg)):
        assert_almost_equal(carg[i], expected[i])


def test_merge_func_code():

    funccode, [pf, pg, ph] = merge_func_code(f, g, h)
    assert_equal(funccode.co_varnames, ('x', 'y', 'z', 'a', 'b', 'c', 'd'))
    exp_pf = [0, 1, 2]
    for i in range(len(pf)): assert_almost_equal(pf[i], exp_pf[i])
    exp_pg = [0, 3, 4]
    for i in range(len(pg)): assert_almost_equal(pg[i], exp_pg[i])
    exp_ph = [0, 5, 6]
    for i in range(len(ph)): assert_almost_equal(ph[i], exp_ph[i])


def test_merge_func_code_prefix():
    funccode, [pf, pg, ph] = merge_func_code(
                                f, g, h,
                                prefix=['f_', 'g_', 'h_'],
                                skip_first=True)
    assert_equal(funccode.co_varnames, ('x', 'f_y', 'f_z',
                                        'g_a', 'g_b', 'h_c', 'h_d'))
    exp_pf = [0, 1, 2]
    for i in range(len(pf)): assert_almost_equal(pf[i], exp_pf[i])
    exp_pg = [0, 3, 4]
    for i in range(len(pg)): assert_almost_equal(pg[i], exp_pg[i])
    exp_ph = [0, 5, 6]
    for i in range(len(ph)): assert_almost_equal(ph[i], exp_ph[i])


def test_merge_func_code_factor_list():
    funccode, [pf, pg, pk_1, pk_2] = merge_func_code(
                                f, g,
                                prefix=['f_', 'g_'],
                                skip_first=True,
                                factor_list=[k_1, k_2])
    assert_equal(funccode.co_varnames, ('x', 'f_y', 'f_z',
                                        'g_a', 'g_b', 'g_i', 'g_j'))
    exp_pf = [0, 1, 2]
    for i in range(len(pf)): assert_almost_equal(pf[i], exp_pf[i])
    exp_pg = [0, 3, 4]
    for i in range(len(pg)): assert_almost_equal(pg[i], exp_pg[i])
    exp_pk_1 = [1, 2]
    for i in range(len(pk_1)): assert_almost_equal(pk_1[i], exp_pk_1[i])
    exp_pk_2 = [5, 6]
    for i in range(len(pk_1)): assert_almost_equal(pk_2[i], exp_pk_2[i])


def test_merge_func_code_skip_prefix():
    funccode, pos = merge_func_code(
                                f, f2,
                                prefix=['f_', 'g_'],
                                skip_first=True,
                                skip_prefix=['z'])
    assert_equal(funccode.co_varnames, ('x', 'f_y', 'z', 'g_a'))


def test_fast_tuple_equal():
    a = (1., 2., 3.)
    b = (1., 2., 3.)
    ret = fast_tuple_equal(a, b, 0)
    assert(ret)

    a = (1., 4., 3.)
    b = (1., 2., 3.)
    ret = fast_tuple_equal(a, b, 0)
    assert(not ret)

    a = (4., 3.)
    b = (1., 4., 3.)
    ret = fast_tuple_equal(a, b, 1)
    assert(ret)

    a = (4., 5.)
    b = (1., 4., 3.)
    ret = fast_tuple_equal(a, b, 1)
    assert(not ret)

    a = tuple([])
    b = tuple([])
    ret = fast_tuple_equal(a, b, 0)
    assert(ret)

