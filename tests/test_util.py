import unittest
from probfit.util import describe, parse_arg

class Func_Code:
    def __init__(self, varname):
        self.co_varnames = varname
        self.co_argcount = len(varname)

class Func1:
    def __init__(self):
        pass
    def __call__(self, x, y):
        return (x - 2.) ** 2 + (y - 5.) ** 2 + 10

class Func2:
    def __init__(self):
        self.func_code = Func_Code(['x', 'y'])
    def __call__(self, *arg):
        return (arg[0] - 2.) ** 2 + (arg[1] - 5.) ** 2 + 10

def func3(x, y):
    return 0.2 * (x - 2.) ** 2 + (y - 5.) ** 2 + 10


def func4(x, y, z):
    return 0.2 * (x - 2.) ** 2 + 0.1 * (y - 5.) ** 2 + 0.25 * (z - 7.) ** 2 + 10

class TestUtil(unittest.TestCase):
    def setUp(self):
        self.f1 = Func1()
        self.f2 = Func2()
        self.f3 = func3

    def iterable_equal(self, x, y):
        self.assertEqual(list(x), list(y))

    def test_parse_arg(self):
        td = {'x':1, 'y':2}
        ts = parse_arg(self.f1, td)
        self.assertEqual(ts, (1, 2))

    def test_describe(self):
        self.iterable_equal(describe(self.f1), ('x', 'y'))
        self.iterable_equal(describe(self.f2), ['x', 'y'])
        self.iterable_equal(describe(self.f3), ('x', 'y'))
