import unittest
from iminuit import Minuit
from probfit import *
from probfit.pdf import  gaussian
from probfit.functor import Extended
from probfit.oneshot import fit_binlh, fit_binx2, fit_uml
import numpy as np
from numpy.random import randn, seed
from math import sqrt
from iminuit.iminuit_warnings import InitialParamWarning
import warnings

def assert_almost_equal(x,y,delta=1e-7):
    if y-delta < x < y+delta:
        pass
    else:
        raise AssertionError('%10f and %10f differ more than %e'%(x,y,delta))

class TestOneshot(unittest.TestCase):

    def setUp(self):
        self.ndata = 20000
        warnings.simplefilter("ignore", InitialParamWarning)
        seed(0)
        self.data = randn(self.ndata)*2. + 5.
        self.wdown = np.empty(self.ndata)
        self.wdown.fill(0.1)

        self.ndata_small = 2000
        self.data_small = randn(self.ndata_small)*2. + 5.

    def test_binx2(self):
        egauss = Extended(gaussian)
        fit,m = fit_binx2(egauss, self.data, bins=100, bound=(1.,9.),
            quiet=True, mean=4., sigma=1., N=10000., print_level=0)
        assert(m.migrad_ok())
        assert_almost_equal(m.values['mean'], 5., delta=3*m.errors['mean'])
        assert_almost_equal(m.values['sigma'], 2., delta=3*m.errors['sigma'])

    def test_binlh(self):
        ngauss=Normalized(gaussian,(1.,9.))
        fit,m = fit_binlh(ngauss, self.data,bins=100, bound=(1.,9.), quiet=True,
            mean=4., sigma=1.5, print_level=0)
        assert(m.migrad_ok())
        assert_almost_equal(m.values['mean'],5.,delta=3*m.errors['mean'])
        assert_almost_equal(m.values['sigma'],2.,delta=3*m.errors['sigma'])

    def test_extended_binlh(self):
        egauss = Extended(gaussian)
        fit,m = fit_binlh(egauss,self.data, bins=100, bound=(1.,9.), quiet=True,
            mean=4., sigma=1., N=10000.,
            print_level=0, extended=True)
        assert(m.migrad_ok())
        assert_almost_equal(m.values['mean'],5.,delta=3*m.errors['mean'])
        assert_almost_equal(m.values['sigma'],2.,delta=3*m.errors['sigma'])
        assert_almost_equal(m.values['N'],20000,delta=3*m.errors['N'])

    def test_extended_binlh_ww(self):
        egauss = Extended(gaussian)
        fit,m = fit_binlh(egauss,self.data,bins=100, bound=(1.,9.), quiet=True,
            mean=4., sigma=1.,N=1000., weights=self.wdown,
            print_level=0, extended=True)
        assert(m.migrad_ok())
        assert_almost_equal(m.values['mean'],5.,delta=m.errors['mean'])
        assert_almost_equal(m.values['sigma'],2.,delta=m.errors['sigma'])
        assert_almost_equal(m.values['N'],2000,delta=m.errors['N'])

    def test_extended_binlh_ww_w2(self):
        egauss = Extended(gaussian)
        fit,m = fit_binlh(egauss,self.data,bins=100, bound=(1.,9.), quiet=True,
            mean=4., sigma=1.,N=1000., weights=self.wdown,
            print_level=0, extended=True)
        assert_almost_equal(m.values['mean'],5.,delta=m.errors['mean'])
        assert_almost_equal(m.values['sigma'],2.,delta=m.errors['sigma'])
        assert_almost_equal(m.values['N'],2000,delta=m.errors['N'])
        assert(m.migrad_ok())

        fit2,m2 = fit_binlh(egauss,self.data,bins=100, bound=(1.,9.), quiet=True,
            mean=4., sigma=1.,N=1000., weights=self.wdown,
            print_level=-1, extended=True, use_w2=True)
        assert_almost_equal(m2.values['mean'],5.,delta=2*m2.errors['mean'])
        assert_almost_equal(m2.values['sigma'],2.,delta=2*m2.errors['sigma'])
        assert_almost_equal(m2.values['N'], 2000., delta=2*m2.errors['N'])
        assert(m2.migrad_ok())

        m.minos()
        m2.minos()

        #now error should scale correctly
        assert_almost_equal( m.errors['mean']/sqrt(10),
                                m2.errors['mean'],
                                delta = m.errors['mean']/sqrt(10)/100.)
        assert_almost_equal(m.errors['sigma']/sqrt(10),
                               m2.errors['sigma'],
                               delta = m.errors['sigma']/sqrt(10)/100.)
        assert_almost_equal(m.errors['N']/sqrt(10),
                               m2.errors['N'],
                               delta = m.errors['N']/sqrt(10)/100.)

    def test_uml(self):
        fit,m = fit_uml(gaussian, self.data, quiet=True,
                        mean=4.5, sigma=1.5, print_level=0)
        assert(m.migrad_ok())
        assert_almost_equal(m.values['mean'],5.,delta=3*m.errors['mean'])
        assert_almost_equal(m.values['sigma'],2.,delta=3*m.errors['sigma'])

    def test_extended_ulh(self):
        eg = Extended(gaussian)
        lh = UnbinnedLH(eg, self.data, extended=True, extended_bound=(-20,20))
        m = Minuit(lh, mean=4.5, sigma=1.5, N=19000., 
                   pedantic=False, print_level=0)
        m.migrad()
        assert_almost_equal(m.values['N'],20000, delta=sqrt(20000.))
        assert(m.migrad_ok())

    def test_extended_ulh_2(self):
        eg = Extended(gaussian)
        lh = UnbinnedLH(eg, self.data, extended=True)
        m = Minuit(lh, mean=4.5, sigma=1.5, N=19000., 
                   pedantic=False, print_level=0)
        m.migrad()
        assert(m.migrad_ok())
        assert_almost_equal(m.values['N'],20000, delta=sqrt(20000.))

if __name__ == '__main__':
    unittest.main()
