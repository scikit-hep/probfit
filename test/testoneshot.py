import unittest
import warnings
from math import sqrt
import numpy as np
from iminuit import Minuit
from iminuit.iminuit_warnings import InitialParamWarning
from probfit import Normalized, Extended, UnbinnedLH
from probfit.pdf import  gaussian
from probfit.oneshot import fit_binlh, fit_binx2, fit_uml

def assert_almost_equal(x, y, delta=1e-7):
    if y - delta < x < y + delta:
        pass
    else:
        raise AssertionError('%10f and %10f differ more than %e' % (x, y, delta))

class TestOneshot(unittest.TestCase):

    def setUp(self):
        self.ndata = 20000
        warnings.simplefilter("ignore", InitialParamWarning)
        np.random.seed(0)
        self.data = np.random.randn(self.ndata) * 2. + 5.
        self.wdown = np.empty(self.ndata)
        self.wdown.fill(0.1)

        self.ndata_small = 2000
        self.data_small = np.random.randn(self.ndata_small) * 2. + 5.

    def test_binx2(self):
        egauss = Extended(gaussian)
        _, minuit = fit_binx2(egauss, self.data, bins=100, bound=(1., 9.),
            quiet=True, mean=4., sigma=1., N=10000., print_level=0)
        assert(minuit.migrad_ok())
        assert_almost_equal(minuit.values['mean'], 5., delta=3 * minuit.errors['mean'])
        assert_almost_equal(minuit.values['sigma'], 2., delta=3 * minuit.errors['sigma'])

    def test_binlh(self):
        ngauss = Normalized(gaussian, (1., 9.))
        _, minuit = fit_binlh(ngauss, self.data, bins=100, bound=(1., 9.), quiet=True,
            mean=4., sigma=1.5, print_level=0)
        assert(minuit.migrad_ok())
        assert_almost_equal(minuit.values['mean'], 5., delta=3 * minuit.errors['mean'])
        assert_almost_equal(minuit.values['sigma'], 2., delta=3 * minuit.errors['sigma'])

    def test_extended_binlh(self):
        egauss = Extended(gaussian)
        _, minuit = fit_binlh(egauss, self.data, bins=100, bound=(1., 9.), quiet=True,
            mean=4., sigma=1., N=10000.,
            print_level=0, extended=True)
        assert(minuit.migrad_ok())
        assert_almost_equal(minuit.values['mean'], 5., delta=3 * minuit.errors['mean'])
        assert_almost_equal(minuit.values['sigma'], 2., delta=3 * minuit.errors['sigma'])
        assert_almost_equal(minuit.values['N'], 20000, delta=3 * minuit.errors['N'])

    def test_extended_binlh_ww(self):
        egauss = Extended(gaussian)
        _, minuit = fit_binlh(egauss, self.data, bins=100, bound=(1., 9.), quiet=True,
            mean=4., sigma=1., N=1000., weights=self.wdown,
            print_level=0, extended=True)
        assert(minuit.migrad_ok())
        assert_almost_equal(minuit.values['mean'], 5., delta=minuit.errors['mean'])
        assert_almost_equal(minuit.values['sigma'], 2., delta=minuit.errors['sigma'])
        assert_almost_equal(minuit.values['N'], 2000, delta=minuit.errors['N'])

    def test_extended_binlh_ww_w2(self):
        egauss = Extended(gaussian)
        _, minuit = fit_binlh(egauss, self.data, bins=100, bound=(1., 9.), quiet=True,
            mean=4., sigma=1., N=1000., weights=self.wdown,
            print_level=0, extended=True)
        assert_almost_equal(minuit.values['mean'], 5., delta=minuit.errors['mean'])
        assert_almost_equal(minuit.values['sigma'], 2., delta=minuit.errors['sigma'])
        assert_almost_equal(minuit.values['N'], 2000, delta=minuit.errors['N'])
        assert(minuit.migrad_ok())

        _, minuit2 = fit_binlh(egauss, self.data, bins=100, bound=(1., 9.), quiet=True,
            mean=4., sigma=1., N=1000., weights=self.wdown,
            print_level= -1, extended=True, use_w2=True)
        assert_almost_equal(minuit2.values['mean'], 5., delta=2 * minuit2.errors['mean'])
        assert_almost_equal(minuit2.values['sigma'], 2., delta=2 * minuit2.errors['sigma'])
        assert_almost_equal(minuit2.values['N'], 2000., delta=2 * minuit2.errors['N'])
        assert(minuit2.migrad_ok())

        minuit.minos()
        minuit2.minos()

        # now error should scale correctly
        assert_almost_equal(minuit.errors['mean'] / sqrt(10),
                            minuit2.errors['mean'],
                            delta=minuit.errors['mean'] / sqrt(10) / 100.)
        assert_almost_equal(minuit.errors['sigma'] / sqrt(10),
                            minuit2.errors['sigma'],
                            delta=minuit.errors['sigma'] / sqrt(10) / 100.)
        assert_almost_equal(minuit.errors['N'] / sqrt(10),
                            minuit2.errors['N'],
                            delta=minuit.errors['N'] / sqrt(10) / 100.)

    def test_uml(self):
        _, minuit = fit_uml(gaussian, self.data, quiet=True,
                         mean=4.5, sigma=1.5, print_level=0)
        assert(minuit.migrad_ok())
        assert_almost_equal(minuit.values['mean'], 5., delta=3 * minuit.errors['mean'])
        assert_almost_equal(minuit.values['sigma'], 2., delta=3 * minuit.errors['sigma'])

    def test_extended_ulh(self):
        eg = Extended(gaussian)
        lh = UnbinnedLH(eg, self.data, extended=True, extended_bound=(-20, 20))
        minuit = Minuit(lh, mean=4.5, sigma=1.5, N=19000.,
                        pedantic=False, print_level=0)
        minuit.migrad()
        assert_almost_equal(minuit.values['N'], 20000, delta=sqrt(20000.))
        assert(minuit.migrad_ok())

    def test_extended_ulh_2(self):
        eg = Extended(gaussian)
        lh = UnbinnedLH(eg, self.data, extended=True)
        minuit = Minuit(lh, mean=4.5, sigma=1.5, N=19000.,
                        pedantic=False, print_level=0)
        minuit.migrad()
        assert(minuit.migrad_ok())
        assert_almost_equal(minuit.values['N'], 20000, delta=sqrt(20000.))
