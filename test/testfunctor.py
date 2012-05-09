import unittest
from dist_fit import *

class TestFunctor(unittest.TestCase):
    def setUp(self):
        pass
    def test_describe_normal_function(self):
        def f(x,y,z):
            return x+y+z
        d = describe(f)
        self.assertEqual(d,tuple(['x','y','z']))

if __name__ == '__main__':
    unittest.main()