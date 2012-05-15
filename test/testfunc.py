import unittest
from dist_fit.cdist_func import *
from dist_fit.util import *
from dist_fit.cdist_fit import *
from math import log
import numpy as np
from numpy.random import randn
class TestFunc(unittest.TestCase):
    def setUp(self):
        self.ndata = 20000
        self.data = randn(self.ndata)*2. + 5.
    
    def test_cruijff(self):
        self.assertEqual(describe(cruijff),tuple(['x', 'm0', 'sigma_L', 'sigma_R', 'alpha_L', 'alpha_R']))
        val = cruijff(0,0,1.,2.,1.,2.)
        self.assertAlmostEqual(val,1.)
        vl = cruijff(0,1,1.,1.,2.,2.)
        vr = cruijff(2,1,1.,1.,2.,2.)
        self.assertAlmostEqual(vl,vr,msg='symmetric test')
        self.assertAlmostEqual(vl,0.7788007830714)
        self.assertAlmostEqual(vr,0.7788007830714)
    
    def test_vectorize_f(self):
        def f(x,y): return x*x+y
        y = 10
        a = np.array([1.,2.,3.])
        expected = [f(x,y) for x in a]
        va = vectorize_f(f,a,tuple([y]))
        for i in range(len(a)): self.assertAlmostEqual(va[i],expected[i])
    
    def test_integrate1d(self):
        def f(x,y):return x*x+y
        def intf(x,y): return x*x*x/3.+y*x
        bound = (-2.,1.)
        y = 3.
        integral = integrate1d(f,bound,1000,tuple([y]))
        analytic = intf(bound[1],y)-intf(bound[0],y)
        self.assertAlmostEqual(integral,analytic,delta=1e-3)
    
    def test_csum(self):
        x = np.array([1,2,3],dtype=np.double)
        s = py_csum(x)
        self.assertAlmostEqual(s,6.)
    
    def test_xlogyx(self):
        def bad(x,y): return x*log(y/x)
        self.assertAlmostEqual(xlogyx(1.,1.),bad(1.,1.))
        self.assertAlmostEqual(xlogyx(1.,2.),bad(1.,2.))
        self.assertAlmostEqual(xlogyx(1.,3.),bad(1.,3.))
        self.assertAlmostEqual(xlogyx(0.,1.),0.)
    
    def test_wlogyx(self):
        def bad(w,y,x): return w*log(y/x)  
        self.assertAlmostEqual(wlogyx(1.,1.,1.),bad(1.,1.,1.))
        self.assertAlmostEqual(wlogyx(1.,2.,3.),bad(1.,2.,3.))
        self.assertAlmostEqual(wlogyx(1e-50,1e-20,1.),bad(1e-50,1e-20,1.))

    def test_construct_arg(self):
        arg = (1,2,3,4,5,6)
        pos = np.array([0,2,4],dtype=np.int)
        carg = construct_arg(arg,pos)
        expected = (1,3,5)
        for i in range(len(carg)):
            self.assertAlmostEqual(carg[i],expected[i])
    
if __name__ == '__main__':
    unittest.main()
