from nose.tools import assert_equal, assert_almost_equal
import numpy as np
from probfit import (describe, rename, Convolve, Normalized,
                     Extended, AddPdf, AddPdfNorm, BlindFunc)
from probfit.pdf import gaussian, ugaussian
from probfit._libstat import integrate1d
from probfit.decorator import extended, normalized

def test_describe_normal_function():
    def f(x, y, z):
        return x + y + z
    d = describe(f)
    assert_equal(list(d), ['x', 'y', 'z'])


def test_Normalized():
    f = ugaussian
    g = Normalized(f, (-1, 1))

    norm = integrate1d(f, (-1., 1.), 1000, (0., 1.))
    assert_almost_equal(g(1., 0., 1.), f(1., 0., 1.) / norm)


def test_normalized_decorator():
    @normalized((-1, 1))
    def f(x, mean, sigma):
        return ugaussian(x, mean, sigma)
    g = Normalized(ugaussian, (-1, 1))
    assert_equal(describe(f), ['x', 'mean', 'sigma'])
    assert_almost_equal(g(1, 0, 1), f(1, 0, 1))


def test_Normalized_cache_hit():
    def f(x, y, z) : return 1.*(x + y + z)
    def g(x, y, z) : return 1.*(x + y + 2 * z)
    nf = Normalized(f, (-10., 10.))
    ng = Normalized(g, (-10., 10.))
    assert_equal(nf.hit, 0)
    nf(1., 2., 3.)
    ng(1., 2., 3.)
    assert_equal(nf.hit, 0)
    nf(3., 2., 3.)
    assert_equal(nf.hit, 1)
    ng(1., 2., 3.)
    assert_equal(ng.hit, 1)


def test_add_pdf():
    def f(x, y, z): return x + y + z
    def g(x, a, b): return 2 * (x + a + b)
    def h(x, c, a): return 3 * (x + c + a)

    A = AddPdf(f, g, h)
    assert_equal(tuple(describe(A)), ('x', 'y', 'z', 'a', 'b', 'c'))

    ret = A(1, 2, 3, 4, 5, 6, 7)
    expected = f(1, 2, 3) + g(1, 4, 5) + h(1, 6, 4)
    assert_almost_equal(ret, expected)

    # wrong integral on purpose
    f.integrate = lambda bound, nint, y, z : 1.  # unbound method works too
    g.integrate = lambda bound, nint, a, b : 2.
    h.integrate = lambda bound, nint, c, a : 3.

    assert_equal(integrate1d(A, (-10., 10.), 100, (1., 2., 3., 4., 5.)), 6.)

def test_add_pdf_factor():
    def f(x, y, z): return x + y + z
    def g(x, a, b): return 2 * (x + a + b)
    def k1(n1, n2): return 3 * (n1 + n2)
    def k2(n1, y): return 4 * (n1 + y) 

    A = AddPdf(f, g, prefix=['f', 'g'], factors=[k1, k2])
    assert_equal(tuple(describe(A)), ('x', 'fy', 'fz', 'ga', 'gb', 'fn1', 'fn2', 'gn1', 'gy'))

    ret = A(1, 2, 3, 4, 5, 6, 7, 8, 9)
    expected = k1(6, 7) * f(1, 2, 3) + k2(8, 9) * g(1, 4, 5)
    assert_almost_equal(ret, expected)

    parts = A.eval_parts(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert_almost_equal(parts[0], k1(6, 7) * f(1, 2, 3))
    assert_almost_equal(parts[1], k2(8, 9) * g(1, 4, 5))


def test_add_pdf_cache():
    def f(x, y, z): return x + y + z
    def g(x, a, b): return 2 * (x + a + b)
    def h(x, c, a): return 3 * (x + c + a)

    A = AddPdf(f, g, h)
    assert_equal(tuple(describe(A)), ('x', 'y', 'z', 'a', 'b', 'c'))

    ret = A(1, 2, 3, 4, 5, 6, 7)
    assert_equal(A.hit, 0)
    expected = f(1, 2, 3) + g(1, 4, 5) + h(1, 6, 4)
    assert_almost_equal(ret, expected)

    ret = A(1, 2, 3, 6, 7, 8, 9)
    assert_equal(A.hit, 1)
    expected = f(1, 2, 3) + g(1, 6, 7) + h(1, 8, 6)
    assert_almost_equal(ret, expected)


def test_extended():
    def f(x, y, z): return x + 2 * y + 3 * z
    g = Extended(f)
    assert_equal(tuple(describe(g)), ('x', 'y', 'z', 'N'))
    assert_equal(g(1, 2, 3, 4), 4 * (f(1, 2, 3)))

    # extended should use analytical when available
    def ana_int(x, y): return y * x ** 2
    ana_int_int = lambda b, n, y: 999.  # wrong on purpose
    ana_int.integrate = ana_int_int
    g = Extended(ana_int)
    assert_almost_equal(g.integrate((0, 1), 100, 5., 2.), 999.*2.)

    # and not fail when it's not available
    def no_ana_int(x, y): return y * x ** 2
    g = Extended(no_ana_int)
    assert_almost_equal(g.integrate((0, 1), 100, 5., 2.), (1.**3) / 3.*5.*2.)

def test_extended_decorator():
    def f(x, y, z): return x + 2 * y + 3 * z

    @extended()
    def g(x, y, z):
        return x + 2 * y + 3 * z

    assert_equal(tuple(describe(g)), ('x', 'y', 'z', 'N'))
    assert_equal(g(1, 2, 3, 4), 4 * (f(1, 2, 3)))


def test_addpdfnorm():
    def f(x, y, z): return x + 2 * y + 3 * z
    def g(x, z, p): return 4 * x + 5 * z + 6 * z
    def p(x, y, q): return 7 * x + 8 * y + 9 * q

    h = AddPdfNorm(f, g)
    assert_equal(describe(h), ['x', 'y', 'z', 'p', 'f_0'])

    q = AddPdfNorm(f, g, p)
    assert_equal(describe(q), ['x', 'y', 'z', 'p', 'q', 'f_0', 'f_1'])

    assert_almost_equal(h(1, 2, 3, 4, 0.1),
            0.1 * f(1, 2, 3) + 0.9 * g(1, 3, 4))

    assert_almost_equal(q(1, 2, 3, 4, 5, 0.1, 0.2),
            0.1 * f(1, 2, 3) + 0.2 * g(1, 3, 4) + 0.7 * p(1, 2, 5))

def test_addpdfnorm_analytical_integrate():
    def f(x, y, z): return x + 2 * y + 3 * z
    def g(x, z, p): return 4 * x + 5 * z + 6 * z
    def p(x, y, q): return 7 * x + 8 * y + 9 * q
    f.integrate = lambda bound, nint, y, z: 1.
    g.integrate = lambda bound, nint, z, p: 2.
    p.integrate = lambda bound, nint, y, q: 3.
    
    q = AddPdfNorm(f, g, p)
    assert_equal(describe(q), ['x', 'y', 'z', 'p', 'q', 'f_0', 'f_1'])

    integral = integrate1d(q, (-10., 10.), 100, (1., 2., 3., 4., 0.1, 0.2))
    assert_almost_equal(integral, 0.1 * 1. + 0.2 * 2. + 0.7 * 3.)
    

def test_convolution():
    f = gaussian
    g = lambda x, mu1, sigma1 : gaussian(x, mu1, sigma1)

    h = Convolve(f, g, (-10, 10), nbins=10000)
    assert_equal(describe(h), ['x', 'mean', 'sigma', 'mu1', 'sigma1'])

    assert_almost_equal(h(1, 0, 1, 1, 2), 0.17839457037411527)  # center
    assert_almost_equal(h(-1, 0, 1, 1, 2), 0.119581456625684)  # left
    assert_almost_equal(h(0, 0, 1, 1, 2), 0.1614180824489487)  # left
    assert_almost_equal(h(2, 0, 1, 1, 2), 0.1614180824489487)  # right
    assert_almost_equal(h(3, 0, 1, 1, 2), 0.119581456625684)  # right


def test_rename():
    def f(x, y, z):
        return None
    assert_equal(describe(f), ['x', 'y', 'z'])
    g = rename(f, ['x', 'a', 'b'])
    assert_equal(describe(g), ['x', 'a', 'b'])

def test_blindfunc():
    np.random.seed(0)
    f = BlindFunc(gaussian, 'mean', 'abcd', width=1.5, signflip=True)
    arg = f.__shift_arg__((1, 1, 1))
    totest = [1., -1.1665264284482637, 1.]
    assert_almost_equal(arg[0], totest[0])
    assert_almost_equal(arg[1], totest[1])
    assert_almost_equal(arg[2], totest[2])
    assert_almost_equal(f.__call__(0.5, 1., 1.), 0.0995003913596)
    np.random.seed(575345)
    f = BlindFunc(gaussian, 'mean', 'abcd', width=1.5, signflip=True)
    arg = f.__shift_arg__((1, 1, 1))
    assert_almost_equal(arg[0], totest[0])
    assert_almost_equal(arg[1], totest[1])
    assert_almost_equal(arg[2], totest[2])
    assert_almost_equal(f.__call__(0.5, 1., 1.), 0.0995003913596)
