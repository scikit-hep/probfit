import unittest
from probfit.costfunc import UnbinnedLH, BinnedLH, BinnedChi2, Chi2Regression,\
                             SimultaneousFit
from probfit.pdf import gaussian, linear
from numpy.random import randn, seed
from math import log,pi,sqrt
import warnings
from iminuit.iminuit_warnings import InitialParamWarning
from probfit.util import describe
from nose.tools import *
from probfit.funcutil import rename
import numpy as np
from iminuit import Minuit

def assert_almost_equal(x,y,delta=1e-7):
    assert(y-delta < x < y+delta)

class TestFit(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", InitialParamWarning)
        seed(0)
        self.ndata = 20000
        self.data = randn(self.ndata)
        self.analytic = self.ndata*0.5*(log(2*pi)+1)

    def test_UnbinnedLH(self):
        f = gaussian
        assert_equal(list(describe(f)), ['x','mean','sigma'])
        lh = UnbinnedLH(gaussian, self.data,)
        assert_equal(list(describe(lh)), ['mean','sigma'])
        assert_almost_equal(lh(0,1), 28188.201229348757)
        m = Minuit(lh)
        assert_equal(m.errordef, 0.5)

    def test_BinnedLH(self):
        #write a better test... this depends on subtraction
        f = gaussian
        assert_equal(list(describe(f)), ['x','mean','sigma'])
        lh = BinnedLH(gaussian, self.data, bound=[-3,3])
        assert_equal(list(describe(lh)), ['mean','sigma'])
        assert_almost_equal(lh(0,1), 20.446130781601543)
        m = Minuit(lh)
        assert_equal(m.errordef, 0.5)


    def test_BinnedChi2(self):
        f = gaussian
        assert_equal(list(describe(f)), ['x','mean','sigma'])
        lh = BinnedChi2(gaussian, self.data, bound=[-3,3])
        assert_equal(list(describe(lh)), ['mean','sigma'])
        assert_almost_equal(lh(0,1), 19951.005399882044)
        m = Minuit(lh)
        assert_equal(m.errordef, 1.0)


    def test_Chi2Regression(self):
        x = np.linspace(1, 10, 10)
        y = 10*x+1
        f = linear
        assert_equal(list(describe(f)), ['x','m','c'])

        lh = Chi2Regression(f, x, y)

        assert_equal(list(describe(lh)), ['m','c'])

        assert_almost_equal(lh(10, 1), 0)

        assert_almost_equal(lh(10, 0), 10.)
        m = Minuit(lh)
        assert_equal(m.errordef, 1.0)


    def test_simultaneous(self):
        seed(0)
        data = randn(10000)
        shifted = data+3.
        g1 = rename(gaussian, ['x', 'lmu', 'sigma'])
        g2 = rename(gaussian, ['x', 'rmu', 'sigma'])
        ulh1 = UnbinnedLH(g1,data)
        ulh2 = UnbinnedLH(g2,shifted)
        sim = SimultaneousFit(ulh1, ulh2)
        assert_equal(describe(sim),['lmu', 'sigma', 'rmu'])
        m = Minuit(sim,sigma=1.2, pedantic=False, print_level=0)
        m.migrad()
        assert_almost_equal(m.values['lmu'], 0., delta=2*m.errors['lmu'])
        assert_almost_equal(m.values['rmu'], 3., delta=2*m.errors['rmu'])
        assert_almost_equal(m.values['sigma'], 1., delta=2*m.errors['sigma'])

if __name__ == '__main__':
    unittest.main()
