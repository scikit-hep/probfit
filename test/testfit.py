import unittest
from probfit.costfunc import UnbinnedLH, BinnedLH, BinnedChi2, Chi2Regression
from probfit.pdf import gaussian, linear
from numpy.random import randn, seed
from math import log,pi,sqrt
import warnings
from iminuit.iminuit_warnings import InitialParamWarning
from probfit.util import describe
import numpy as np

class TestFit(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore", InitialParamWarning)
        seed(0)
        self.ndata = 20000
        self.data = randn(self.ndata)
        self.analytic = self.ndata*0.5*(log(2*pi)+1)

    def test_UnbinnedLH(self):
        f = gaussian
        self.assertEqual(list(describe(f)), ['x','mean','sigma'])
        lh = UnbinnedLH(gaussian, self.data,)
        self.assertEqual(list(describe(lh)), ['mean','sigma'])
        self.assertAlmostEqual(lh(0,1), self.analytic,
                                delta=self.analytic*0.01)

    def test_BinnedLH(self):
        #write a better test... this depends on subtraction
        f = gaussian
        self.assertEqual(list(describe(f)), ['x','mean','sigma'])
        lh = BinnedLH(gaussian, self.data, bound=[-3,3])
        self.assertEqual(list(describe(lh)), ['mean','sigma'])
        self.assertAlmostEqual(lh(0,1), 20.446130781601543)


    def test_BinnedChi2(self):
        f = gaussian
        self.assertEqual(list(describe(f)), ['x','mean','sigma'])
        lh = BinnedChi2(gaussian, self.data, bound=[-3,3])
        self.assertEqual(list(describe(lh)), ['mean','sigma'])
        self.assertAlmostEqual(lh(0,1), 19951.005399882044)


    def test_Chi2Regression(self):
        x = np.linspace(1, 10, 10)
        y = 10*x+1
        f = linear
        self.assertEqual(list(describe(f)), ['x','m','c'])

        lh = Chi2Regression(f, x, y)

        self.assertEqual(list(describe(lh)), ['m','c'])

        self.assertAlmostEqual(lh(10, 1), 0)

        self.assertAlmostEqual(lh(10, 0), 10.)

if __name__ == '__main__':
    unittest.main()
