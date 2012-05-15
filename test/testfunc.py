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
        #print carg
        for i in range(len(carg)):
            self.assertAlmostEqual(carg[i],expected[i])
        # 
        # def test_tuple2array(self):
        #     t = (1.,2.,3.)
        # 
        #     a = tuple2array(t,3,0)
        #     for i in range(len(t)):
        #         self.assertAlmostEqual(a[i],t[i])
        #     
        #     expected = [2.,3.]
        #     a = tuple2array(t,3,1)
        #     for i in range(len(a)):
        #         self.assertAlmostEqual(a[i],expected[i])
        #     self.assertEqual(len(a),2)
        #     
    def test_merge_func_code(self):
        def f(x,y,z): return x+y+z
        def g(x,a,b): return x+a+b
        def h(x,c,d): return x+c+d
        
        funccode, [pf,pg,ph] = merge_func_code(f,g,h)
        self.assertEqual(funccode.co_varnames,('x','y','z','a','b','c','d'))
        exp_pf = [0,1,2]
        for i in range(len(pf)): self.assertAlmostEqual(pf[i],exp_pf[i])
        exp_pg = [0,3,4]
        for i in range(len(pf)): self.assertAlmostEqual(pg[i],exp_pg[i])
        exp_ph = [0,5,6]
        for i in range(len(pf)): self.assertAlmostEqual(ph[i],exp_ph[i])
        
        funccode, [pf,pg,ph] = merge_func_code(f,g,h,prefix=['f_','g_','h_'])
        self.assertEqual(funccode.co_varnames,('x','f_y','f_z','g_a','g_b','h_c','h_d'))
        exp_pf = [0,1,2]
        for i in range(len(pf)): self.assertAlmostEqual(pf[i],exp_pf[i])
        exp_pg = [0,3,4]
        for i in range(len(pf)): self.assertAlmostEqual(pg[i],exp_pg[i])
        exp_ph = [0,5,6]
        for i in range(len(pf)): self.assertAlmostEqual(ph[i],exp_ph[i])

    def test_add_pdf(self):
        def f(x,y,z): return x+y+z
        def g(x,a,b): return 2*(x+a+b)
        def h(x,c,d): return 3*(x+c+d)
        
        A = AddPdf(f,g,h)
        self.assertEqual(describe(A),('x','y','z','a','b','c','d'))
        
        ret = A(1,2,3,4,5,6,7)
        expected = f(1,2,3)+g(1,4,5)+h(1,6,7)
        self.assertAlmostEqual(ret,expected)

    def test_add_pdf_cache(self):
        def f(x,y,z): return x+y+z
        def g(x,a,b): return 2*(x+a+b)
        def h(x,c,d): return 3*(x+c+d)

        A = AddPdf(f,g,h)
        self.assertEqual(describe(A),('x','y','z','a','b','c','d'))

        ret = A(1,2,3,4,5,6,7)
        self.assertEqual(A.hit,0)
        expected = f(1,2,3)+g(1,4,5)+h(1,6,7)
        self.assertAlmostEqual(ret,expected)

        ret = A(1,2,3,6,7,8,9)
        self.assertEqual(A.hit,1)
        expected = f(1,2,3)+g(1,6,7)+h(1,8,9)        
        self.assertAlmostEqual(ret,expected)
        
    def test_fast_tuple_equal(self):
        a = (1.,2.,3.)
        b = (1.,2.,3.)
        ret = fast_tuple_equal(a,b,0)
        self.assertTrue(ret)
        
        a = (1.,4.,3.)
        b = (1.,2.,3.)
        ret = fast_tuple_equal(a,b,0)
        self.assertFalse(ret)

        a = (4.,3.)
        b = (1.,4.,3.)
        ret = fast_tuple_equal(a,b,1)
        self.assertTrue(ret)
        
        a = (4.,5.)
        b = (1.,4.,3.)
        ret = fast_tuple_equal(a,b,1)
        self.assertFalse(ret)
        
    def test_Normalize_cache_hit(self):
        def f(x,y,z) : return 1.*(x+y+z)
        def g(x,y,z) : return 1.*(x+y+2*z)
        nf = Normalize(f,(-10.,10.))
        ng = Normalize(g,(-10.,10.))
        self.assertEqual(nf.hit,0)
        nf(1.,2.,3.)
        ng(1.,2.,3.)
        self.assertEqual(nf.hit,0)
        nf(3.,2.,3.)
        self.assertEqual(nf.hit,1)
        ng(1.,2.,3.)
        self.assertEqual(ng.hit,1)
        
    
if __name__ == '__main__':
    unittest.main()
