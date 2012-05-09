import unittest
from dist_fit import *

class TestFunc(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_cruijff(self):
        self.assertEqual(describe(cruijff),tuple(['x', 'm0', 'sigma_L', 'sigma_R', 'alpha_L', 'alpha_R']))
        val = cruijff(0,0,1.,2.,1.,2.)
        self.assertAlmostEqual(val,1.)
        vl = cruijff(0,1,1.,1.,2.,2.)
        vr = cruijff(2,1,1.,1.,2.,2.)
        self.assertAlmostEqual(vl,vr,msg='symmetric test')
        self.assertAlmostEqual(vl,0.7788007830714)
        self.assertAlmostEqual(vr,0.7788007830714)
