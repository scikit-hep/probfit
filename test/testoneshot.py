import unittest
from probfit import *
from probfit.cdist_func import Extend, gaussian
import numpy as np
from numpy.random import randn, seed
from math import sqrt
class TestOneshot(unittest.TestCase):

    def setUp(self):
        self.ndata = 20000
        seed(0)
        self.data = randn(self.ndata)*2. + 5.
        self.wdown = np.empty(self.ndata)
        self.wdown.fill(0.1)

        self.ndata_small = 2000
        self.data_small = randn(self.ndata_small)*2. + 5.

    def test_binx2(self):
        egauss = Extend(gaussian)
        fit,m = fit_binx2(egauss, self.data, bins=100, range=(1.,9.),
            quiet=True, mean=4., sigma=1., N=10000., print_level=0)
        self.assertAlmostEqual(m.values['mean'], 5., delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'], 2., delta=3*m.errors['sigma'])

    def test_binlh(self):
        ngauss=Normalized(gaussian,(1.,9.))
        fit,m = fit_binlh(ngauss, self.data,bins=1000, range=(1.,9.), quiet=True,
            mean=4., sigma=1.5, print_level=0)
        self.assertAlmostEqual(m.values['mean'],5.,delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'],2.,delta=3*m.errors['sigma'])

    def test_extended_binlh(self):
        egauss = Extend(gaussian)
        fit,m = fit_binlh(egauss,self.data, bins=1000, range=(1.,9.), quiet=True,
            mean=4., sigma=1., N=10000.,
            print_level=0, extended=True)
        self.assertAlmostEqual(m.values['mean'],5.,delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'],2.,delta=3*m.errors['sigma'])
        self.assertAlmostEqual(m.values['N'],20000,delta=3*m.errors['N'])

    def test_extended_binlh_ww(self):
        egauss = Extend(gaussian)
        fit,m = fit_binlh(egauss,self.data,bins=1000, range=(1.,9.), quiet=True,
            mean=4., sigma=1.,N=1000., weights=self.wdown,
            print_level=-1, extended=True)
        self.assertAlmostEqual(m.values['mean'],5.,delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'],2.,delta=3*m.errors['sigma'])
        self.assertAlmostEqual(m.values['N'],2000,delta=3*m.errors['N'])

    def test_extended_binlh_ww_w2(self):
        egauss = Extend(gaussian)
        fit,m = fit_binlh(egauss,self.data,bins=1000, range=(1.,9.), quiet=True,
            mean=4., sigma=1.,N=1000., weights=self.wdown,
            print_level=-1, extended=True)
        self.assertAlmostEqual(m.values['mean'],5.,delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'],2.,delta=3*m.errors['sigma'])
        self.assertAlmostEqual(m.values['N'],2000,delta=3*m.errors['N'])

        fit2,m2 = fit_binlh(egauss,self.data,bins=1000, range=(1.,9.), quiet=True,
            mean=4., sigma=1.,N=1000., weights=self.wdown,
            print_level=-1, extended=True, use_w2=True)
        #self.assertAlmostEqual(m2.values['mean'],5.,delta=3*m2.errors['mean'])
        self.assertAlmostEqual(m2.values['sigma'],2.,delta=3*m2.errors['sigma'])
        self.assertAlmostEqual(m2.values['N'], 2000., delta=3*m2.errors['N'])
        m.minos()
        m2.minos()

        #now error should scale correctly
        self.assertAlmostEqual(m.errors['mean']/sqrt(10),m2.errors['mean'],delta = m.errors['mean']/sqrt(10))
        self.assertAlmostEqual(m.errors['sigma']/sqrt(10),m2.errors['sigma'],delta = m.errors['sigma']/sqrt(10))
        self.assertAlmostEqual(m.errors['N']/sqrt(10),m2.errors['N'],delta = m.errors['N']/sqrt(10))

    def test_gen_toy(self):
        pdf = gaussian
        toy = gen_toy(pdf,10000,(-5,5),mean=0,sigma=1)
        binlh = BinnedLH(pdf,toy,range=(-5,5),bins=100)
        lh = binlh(0.,1.)
        for x in toy:
            self.assertLessEqual(x,5)
            self.assertGreaterEqual(x,-5)
        self.assertEqual(len(toy),10000)
        self.assertLess(lh/100.,1.)


    def test_uml(self):
        fit,m = fit_uml(gaussian, self.data, quiet=True,
                        mean=4.5, sigma=1.5, print_level=0)
        self.assertAlmostEqual(m.values['mean'],5.,delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'],2.,delta=3*m.errors['sigma'])

if __name__ == '__main__':
    unittest.main()
