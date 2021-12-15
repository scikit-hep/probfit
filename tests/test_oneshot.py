import warnings
from math import sqrt

import numpy as np
from iminuit import Minuit
from iminuit.iminuit_warnings import InitialParamWarning
from numpy.testing import assert_allclose

from probfit import Extended, Normalized, UnbinnedLH
from probfit.oneshot import fit_binlh, fit_binx2, fit_uml
from probfit.pdf import gaussian


class TestOneshot:
    def setup(self):
        self.ndata = 20000
        warnings.simplefilter("ignore", InitialParamWarning)
        np.random.seed(0)
        self.data = np.random.randn(self.ndata) * 2.0 + 5.0
        self.wdown = np.empty(self.ndata)
        self.wdown.fill(0.1)

        self.ndata_small = 2000
        self.data_small = np.random.randn(self.ndata_small) * 2.0 + 5.0

    def test_binx2(self):
        egauss = Extended(gaussian)
        _, minuit = fit_binx2(
            egauss,
            self.data,
            bins=100,
            bound=(1.0, 9.0),
            quiet=True,
            mean=4.0,
            sigma=1.0,
            N=10000.0,
            print_level=0,
        )
        assert minuit.migrad_ok()
        assert_allclose(minuit.values["mean"], 5.0, atol=3 * minuit.errors["mean"])
        assert_allclose(minuit.values["sigma"], 2.0, atol=3 * minuit.errors["sigma"])

    def test_binlh(self):
        ngauss = Normalized(gaussian, (1.0, 9.0))
        _, minuit = fit_binlh(
            ngauss,
            self.data,
            bins=100,
            bound=(1.0, 9.0),
            quiet=True,
            mean=4.0,
            sigma=1.5,
            print_level=0,
        )
        assert minuit.migrad_ok()
        assert_allclose(minuit.values["mean"], 5.0, atol=3 * minuit.errors["mean"])
        assert_allclose(minuit.values["sigma"], 2.0, atol=3 * minuit.errors["sigma"])

    def test_extended_binlh(self):
        egauss = Extended(gaussian)
        _, minuit = fit_binlh(
            egauss,
            self.data,
            bins=100,
            bound=(1.0, 9.0),
            quiet=True,
            mean=4.0,
            sigma=1.0,
            N=10000.0,
            print_level=0,
            extended=True,
        )
        assert minuit.migrad_ok()
        assert_allclose(minuit.values["mean"], 5.0, atol=3 * minuit.errors["mean"])
        assert_allclose(minuit.values["sigma"], 2.0, atol=3 * minuit.errors["sigma"])
        assert_allclose(minuit.values["N"], 20000, atol=3 * minuit.errors["N"])

    def test_extended_binlh_ww(self):
        egauss = Extended(gaussian)
        _, minuit = fit_binlh(
            egauss,
            self.data,
            bins=100,
            bound=(1.0, 9.0),
            quiet=True,
            mean=4.0,
            sigma=1.0,
            N=1000.0,
            weights=self.wdown,
            print_level=0,
            extended=True,
        )
        assert minuit.migrad_ok()
        assert_allclose(minuit.values["mean"], 5.0, atol=minuit.errors["mean"])
        assert_allclose(minuit.values["sigma"], 2.0, atol=minuit.errors["sigma"])
        assert_allclose(minuit.values["N"], 2000, atol=minuit.errors["N"])

    def test_extended_binlh_ww_w2(self):
        egauss = Extended(gaussian)
        _, minuit = fit_binlh(
            egauss,
            self.data,
            bins=100,
            bound=(1.0, 9.0),
            quiet=True,
            mean=4.0,
            sigma=1.0,
            N=1000.0,
            weights=self.wdown,
            print_level=0,
            extended=True,
        )
        assert_allclose(minuit.values["mean"], 5.0, atol=minuit.errors["mean"])
        assert_allclose(minuit.values["sigma"], 2.0, atol=minuit.errors["sigma"])
        assert_allclose(minuit.values["N"], 2000, atol=minuit.errors["N"])
        assert minuit.migrad_ok()

        _, minuit2 = fit_binlh(
            egauss,
            self.data,
            bins=100,
            bound=(1.0, 9.0),
            quiet=True,
            mean=4.0,
            sigma=1.0,
            N=1000.0,
            weights=self.wdown,
            print_level=-1,
            extended=True,
            use_w2=True,
        )
        assert_allclose(minuit2.values["mean"], 5.0, atol=2 * minuit2.errors["mean"])
        assert_allclose(minuit2.values["sigma"], 2.0, atol=2 * minuit2.errors["sigma"])
        assert_allclose(minuit2.values["N"], 2000.0, atol=2 * minuit2.errors["N"])
        assert minuit2.migrad_ok()

        minuit.minos()
        minuit2.minos()

        # now error should scale correctly
        assert_allclose(
            minuit.errors["mean"] / sqrt(10),
            minuit2.errors["mean"],
            atol=minuit.errors["mean"] / sqrt(10) / 100.0,
        )
        assert_allclose(
            minuit.errors["sigma"] / sqrt(10),
            minuit2.errors["sigma"],
            atol=minuit.errors["sigma"] / sqrt(10) / 100.0,
        )
        assert_allclose(
            minuit.errors["N"] / sqrt(10),
            minuit2.errors["N"],
            atol=minuit.errors["N"] / sqrt(10) / 100.0,
        )

    def test_uml(self):
        _, minuit = fit_uml(
            gaussian, self.data, quiet=True, mean=4.5, sigma=1.5, print_level=0
        )
        assert minuit.migrad_ok()
        assert_allclose(minuit.values["mean"], 5.0, atol=3 * minuit.errors["mean"])
        assert_allclose(minuit.values["sigma"], 2.0, atol=3 * minuit.errors["sigma"])

    def test_extended_ulh(self):
        eg = Extended(gaussian)
        lh = UnbinnedLH(eg, self.data, extended=True, extended_bound=(-20, 20))
        minuit = Minuit(
            lh, mean=4.5, sigma=1.5, N=19000.0, pedantic=False, print_level=0
        )
        minuit.migrad()
        assert_allclose(minuit.values["N"], 20000, atol=sqrt(20000.0))
        assert minuit.migrad_ok()

    def test_extended_ulh_2(self):
        eg = Extended(gaussian)
        lh = UnbinnedLH(eg, self.data, extended=True)
        minuit = Minuit(
            lh, mean=4.5, sigma=1.5, N=19000.0, pedantic=False, print_level=0
        )
        minuit.migrad()
        assert minuit.migrad_ok()
        assert_allclose(minuit.values["N"], 20000, atol=sqrt(20000.0))
