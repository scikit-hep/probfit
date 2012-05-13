import unittest
from dist_fit import *
import numpy as np
from numpy.random import randn
class TestOneshot(unittest.TestCase):
    
    def setUp(self):
        self.ndata = 200000
        self.data = randn(self.ndata)*2. + 5.
        pass

    def test_binx2(self):
        egauss = Extend(gaussian)
        fit,m = fit_binx2(egauss,self.data,bins=100, range=(1.,9.), quiet=True, mean=4., sigma=1.,N=10000.,printlevel=-1)
        self.assertAlmostEqual(m.values['mean'],5.,delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'],2.,delta=3*m.errors['sigma'])
    
    def test_binlh(self):
        ngauss=Normalize(gaussian,(1.,9.))
        fit,m = fit_binlh(ngauss,self.data,bins=1000, range=(1.,9.), quiet=True, mean=4., sigma=1.5,printlevel=-1,f_verbose=False)
        self.assertAlmostEqual(m.values['mean'],5.,delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'],2.,delta=3*m.errors['sigma'])
    
    def test_extended_binlh(self):
        egauss = Extend(gaussian)
        fit,m = fit_binlh(egauss,self.data,bins=1000, range=(1.,9.), quiet=True, 
            mean=4., sigma=1.,N=20000.,
            printlevel=-1,f_verbose=False, extended=True)
        self.assertAlmostEqual(m.values['mean'],5.,delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'],2.,delta=3*m.errors['sigma'])
    
    def test_uml(self):
        fit,m = fit_uml(gaussian,self.data, quiet=True, mean=4.5, sigma=1.5,printlevel=-1)
        self.assertAlmostEqual(m.values['mean'],5.,delta=3*m.errors['mean'])
        self.assertAlmostEqual(m.values['sigma'],2.,delta=3*m.errors['sigma'])

if __name__ == '__main__':
    unittest.main()