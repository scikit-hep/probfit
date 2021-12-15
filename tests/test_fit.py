import warnings

import iminuit
import numpy as np
from iminuit import describe
from iminuit.iminuit_warnings import InitialParamWarning
from numpy.testing import assert_allclose

from probfit.costfunc import (
    BinnedChi2,
    BinnedLH,
    Chi2Regression,
    SimultaneousFit,
    UnbinnedLH,
)
from probfit.funcutil import rename
from probfit.pdf import gaussian, linear


class TestFit:
    def setup(self):
        warnings.simplefilter("ignore", InitialParamWarning)
        np.random.seed(0)
        self.ndata = 20000
        self.data = np.random.randn(self.ndata)
        self.analytic = self.ndata * 0.5 * (np.log(2 * np.pi) + 1)

    def test_UnbinnedLH(self):
        f = gaussian
        assert list(describe(f)) == ["x", "mean", "sigma"]
        lh = UnbinnedLH(
            gaussian,
            self.data,
        )
        assert list(describe(lh)) == ["mean", "sigma"]
        assert_allclose(lh(0, 1), 28188.201229348757)
        minuit = iminuit.Minuit(lh)
        assert_allclose(minuit.errordef, 0.5)

    def test_BinnedLH(self):
        # write a better test... this depends on subtraction
        f = gaussian
        assert list(describe(f)) == ["x", "mean", "sigma"]
        lh = BinnedLH(gaussian, self.data, bound=[-3, 3])
        assert list(describe(lh)) == ["mean", "sigma"]
        assert_allclose(lh(0, 1), 20.446130781601543, atol=1)
        minuit = iminuit.Minuit(lh)
        assert_allclose(minuit.errordef, 0.5)

    def test_BinnedChi2(self):
        f = gaussian
        assert list(describe(f)) == ["x", "mean", "sigma"]
        lh = BinnedChi2(gaussian, self.data, bound=[-3, 3])
        assert list(describe(lh)) == ["mean", "sigma"]
        assert_allclose(lh(0, 1), 19951.005399882044, atol=1)
        minuit = iminuit.Minuit(lh)
        assert_allclose(minuit.errordef, 1.0)

    def test_Chi2Regression(self):
        x = np.linspace(1, 10, 10)
        y = 10 * x + 1
        f = linear
        assert list(describe(f)) == ["x", "m", "c"]

        lh = Chi2Regression(f, x, y)

        assert list(describe(lh)) == ["m", "c"]

        assert_allclose(lh(10, 1), 0)

        assert_allclose(lh(10, 0), 10.0)
        minuit = iminuit.Minuit(lh)
        assert_allclose(minuit.errordef, 1.0)

    def test_simultaneous(self):
        np.random.seed(0)
        data = np.random.randn(10000)
        shifted = data + 3.0
        g1 = rename(gaussian, ["x", "lmu", "sigma"])
        g2 = rename(gaussian, ["x", "rmu", "sigma"])
        ulh1 = UnbinnedLH(g1, data)
        ulh2 = UnbinnedLH(g2, shifted)
        sim = SimultaneousFit(ulh1, ulh2)
        assert describe(sim) == ["lmu", "sigma", "rmu"]
        minuit = iminuit.Minuit(sim, sigma=1.2, pedantic=False, print_level=0)
        minuit.migrad()
        assert minuit.migrad_ok()
        assert_allclose(minuit.values["lmu"], 0.0, atol=2 * minuit.errors["lmu"])
        assert_allclose(minuit.values["rmu"], 3.0, atol=2 * minuit.errors["rmu"])
        assert_allclose(minuit.values["sigma"], 1.0, atol=2 * minuit.errors["sigma"])
