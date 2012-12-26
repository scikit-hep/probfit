import unittest
from probfit import *
from probfit._libstat import integrate1d
from nose.tools import *

def test_describe_normal_function():
    def f(x,y,z):
        return x+y+z
    d = describe(f)
    assert_equal(list(d),['x','y','z'])

def test_Normalize():
    f = ugaussian
    g = Normalized(f,(-1,1))

    norm = integrate1d(f,(-1.,1.),1000,(0.,1.))
    assert_almost_equal(g(1.,0.,1.),f(1.,0.,1.)/norm)

def test_Normalize_cache_hit():
    def f(x,y,z) : return 1.*(x+y+z)
    def g(x,y,z) : return 1.*(x+y+2*z)
    nf = Normalized(f,(-10.,10.))
    ng = Normalized(g,(-10.,10.))
    assert_equal(nf.hit,0)
    nf(1.,2.,3.)
    ng(1.,2.,3.)
    assert_equal(nf.hit,0)
    nf(3.,2.,3.)
    assert_equal(nf.hit,1)
    ng(1.,2.,3.)
    assert_equal(ng.hit,1)

def test_add_pdf():
    def f(x,y,z): return x+y+z
    def g(x,a,b): return 2*(x+a+b)
    def h(x,c,d): return 3*(x+c+d)

    A = AddPdf(f,g,h)
    assert_equal(tuple(describe(A)),('x','y','z','a','b','c','d'))

    ret = A(1,2,3,4,5,6,7)
    expected = f(1,2,3)+g(1,4,5)+h(1,6,7)
    assert_almost_equal(ret,expected)

def test_add_pdf_cache():
    def f(x,y,z): return x+y+z
    def g(x,a,b): return 2*(x+a+b)
    def h(x,c,d): return 3*(x+c+d)

    A = AddPdf(f,g,h)
    assert_equal(tuple(describe(A)), ('x','y','z','a','b','c','d'))

    ret = A(1,2,3,4,5,6,7)
    assert_equal(A.hit,0)
    expected = f(1,2,3)+g(1,4,5)+h(1,6,7)
    assert_almost_equal(ret,expected)

    ret = A(1,2,3,6,7,8,9)
    assert_equal(A.hit,1)
    expected = f(1,2,3)+g(1,6,7)+h(1,8,9)
    assert_almost_equal(ret,expected)
