"""Test that output figures look sensible.

These tests use pytest-mpl [1] to compare the figures created in the test_*
methods with those in the baseline/ directory (relative to this file).
To generate the baseline figures, run:

    py.test --mpl-generate-path=baseline tests/_test_plotting.py

This will put the figures in the baseline/ directory relative to where the
tests were run. You can then copy the ones that you want to update into the
tests/baseline directory.

[1]: https://pypi.python.org/pypi/pytest-mpl
"""
import pytest
import numpy as np
from matplotlib import pyplot as plt
from iminuit import Minuit
import sys
from probfit.plotting import draw_pdf, draw_compare_hist
from probfit.pdf import gaussian, linear
from probfit.funcutil import rename
from probfit.functor import Extended, AddPdfNorm, AddPdf
from probfit.costfunc import UnbinnedLH, BinnedLH, BinnedChi2, Chi2Regression, \
    SimultaneousFit

major, minor = sys.version_info[0:2]
if major >= 3 and minor >= 5:
    special_tol = 60.0
else:
    special_tol = 2.0

def image_comparison(filename, **kwargs):
    """Decorator to provide a new Figure instance and return it.

    This allows the mpl_image_compare wrapper to be used seamlessly: methods
    wrapped in this decorator can just draw with `plt.whatever`, and then
    mpl_image_compare with use plt.gcf (the global Figure instance) to compare
    to the baseline.
    """
    def wrapper(func):
        def wrapped():
            fig = plt.figure()
            func()
            return fig
        return pytest.mark.mpl_image_compare(filename=filename, **kwargs)(wrapped)
    return wrapper


@image_comparison('draw_pdf.png')
def test_draw_pdf():
    f = gaussian
    draw_pdf(f, {'mean': 1., 'sigma': 2.}, bound=(-10, 10))

    
@image_comparison('draw_pdf_linear.png')
def test_draw_pdf_linear():
    f = linear
    draw_pdf(f, {'m': 1., 'c': 2.}, bound=(-10, 10))


# There is a slight difference in the x-axis tick label positioning for this
# plot between Python 2 and 3, it's not important here so increase the RMS
# slightly such that it's ignored
@image_comparison('draw_compare_hist_gaussian.png', tolerance=2.05)
def test_draw_compare_hist():
    np.random.seed(0)
    data = np.random.randn(10000)
    f = gaussian
    draw_compare_hist(f, {'mean': 0., 'sigma': 1.}, data, normed=True)


# There is a slight difference in the x-axis tick label positioning for this
# plot between Python 2 and 3, it's not important here so increase the RMS
# slightly such that it's ignored
@image_comparison('draw_compare_hist_no_norm.png', tolerance=2.05)
def test_draw_compare_hist_no_norm():
    np.random.seed(0)
    data = np.random.randn(10000)
    f = Extended(gaussian)
    draw_compare_hist(f, {'mean': 0., 'sigma': 1., 'N': 10000}, data, normed=False)


@image_comparison('draw_ulh.png')
def test_draw_ulh():
    np.random.seed(0)
    data = np.random.randn(1000)
    ulh = UnbinnedLH(gaussian, data)
    ulh.draw(args=(0., 1.))


@image_comparison('draw_ulh_extend.png')
def test_draw_ulh_extend():
    np.random.seed(0)
    data = np.random.randn(1000)
    ulh = UnbinnedLH(Extended(gaussian), data, extended=True)
    ulh.draw(args=(0., 1., 1000))


@image_comparison('draw_residual_ulh.png')
def test_draw_residual_ulh():
    np.random.seed(0)
    data = np.random.randn(1000)
    ulh = UnbinnedLH(gaussian, data)
    ulh.draw_residual(args=(0., 1.))


@image_comparison('draw_residual_ulh_norm.png')
def test_draw_residual_ulh_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    ulh = UnbinnedLH(gaussian, data)
    ulh.draw_residual(args=(0., 1.), norm=True)
    plt.ylim(-7., 3.)
    plt.xlim(-4., 3.)


@image_comparison('draw_residual_ulh_norm_no_errbars.png', tolerance=special_tol)
def test_draw_residual_ulh_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    ulh = UnbinnedLH(gaussian, data)
    ulh.draw_residual(args=(0., 1.), norm=True, show_errbars=False)


@image_comparison('draw_residual_ulh_norm_options.png')
def test_draw_residual_ulh_norm_options():
    np.random.seed(0)
    data = np.random.randn(1000)
    ulh = UnbinnedLH(gaussian, data)
    ulh.draw_residual(args=(0., 1.), norm=True, color='green', capsize=2,
                      grid=False, zero_line=False)


@image_comparison('draw_ulh_extend_residual_norm.png')
def test_draw_ulh_extend_residual_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    ulh = UnbinnedLH(Extended(gaussian), data, extended=True)
    ulh.draw_residual(args=(0., 1., 1000), norm=True)
    plt.ylim(-7.,3.)


@image_comparison('draw_ulh_with_minuit.png')
def test_draw_ulh_with_minuit():
    np.random.seed(0)
    data = np.random.randn(1000)
    ulh = UnbinnedLH(gaussian, data)
    minuit = Minuit(ulh, mean=0, sigma=1)
    ulh.draw(minuit)


@image_comparison('draw_blh.png')
def test_draw_blh():
    np.random.seed(0)
    data = np.random.randn(1000)
    blh = BinnedLH(gaussian, data)
    blh.draw(args=(0., 1.))


@image_comparison('draw_blh_extend.png')
def test_draw_blh_extend():
    np.random.seed(0)
    data = np.random.randn(1000)
    blh = BinnedLH(Extended(gaussian), data, extended=True)
    blh.draw(args=(0., 1., 1000))


@image_comparison('draw_residual_blh.png')
def test_draw_residual_blh():
    np.random.seed(0)
    data = np.random.randn(1000)
    blh = BinnedLH(gaussian, data)
    blh.draw_residual(args=(0., 1.))


@image_comparison('draw_residual_blh_norm.png')
def test_draw_residual_blh_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    blh = BinnedLH(gaussian, data)
    blh.draw_residual(args=(0., 1.), norm=True)
    plt.ylim(-4., 3.)
    plt.xlim(-4., 3.)


@image_comparison('draw_residual_blh_norm_options.png')
def test_draw_residual_blh_norm_options():
    np.random.seed(0)
    data = np.random.randn(1000)
    blh = BinnedLH(gaussian, data)
    blh.draw_residual(args=(0., 1.), norm=True, color='green', capsize=2,
                      grid=False, zero_line=False)


@image_comparison('draw_residual_blh_norm_no_errbars.png', tolerance=special_tol)
def test_draw_residual_blh_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    blh = BinnedLH(gaussian, data)
    blh.draw_residual(args=(0., 1.), norm=True, show_errbars=False)


@image_comparison('draw_blh_extend_residual_norm.png')
def test_draw_blh_extend_residual_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    blh = BinnedLH(Extended(gaussian), data, extended=True)
    blh.draw_residual(args=(0., 1., 1000), norm=True)


@image_comparison('draw_bx2.png')
def test_draw_bx2():
    np.random.seed(0)
    data = np.random.randn(1000)
    blh = BinnedChi2(Extended(gaussian), data)
    blh.draw(args=(0., 1., 1000))


@image_comparison('draw_x2reg.png')
def test_draw_x2reg():
    np.random.seed(0)
    x = np.linspace(0, 1, 100)
    y = 10. * x + np.random.randn(100)
    err = np.array([1] * 100)
    blh = Chi2Regression(linear, x, y, err)
    blh.draw(args=(10., 0.))


@image_comparison('draw_ulh_with_parts.png')
def test_ulh_with_parts():
    np.random.seed(0)
    data = np.random.randn(10000)
    shifted = data + 3.
    data = np.append(data, [shifted])
    g1 = rename(gaussian, ['x', 'lmu', 'lsigma'])
    g2 = rename(gaussian, ['x', 'rmu', 'rsigma'])
    allpdf = AddPdfNorm(g1, g2)
    ulh = UnbinnedLH(allpdf, data)
    ulh.draw(args=(0, 1, 3, 1, 0.5), parts=True)


@image_comparison('draw_blh_with_parts.png')
def test_blh_with_parts():
    np.random.seed(0)
    data = np.random.randn(10000)
    shifted = data + 3.
    data = np.append(data, [shifted])
    g1 = rename(gaussian, ['x', 'lmu', 'lsigma'])
    g2 = rename(gaussian, ['x', 'rmu', 'rsigma'])
    allpdf = AddPdfNorm(g1, g2)
    blh = BinnedLH(allpdf, data)
    blh.draw(args=(0, 1, 3, 1, 0.5), parts=True)


@image_comparison('draw_bx2_with_parts.png')
def test_bx2_with_parts():
    np.random.seed(0)
    data = np.random.randn(10000)
    shifted = data + 3.
    data = np.append(data, [shifted])
    g1 = Extended(rename(gaussian, ['x', 'lmu', 'lsigma']), extname='N1')
    g2 = Extended(rename(gaussian, ['x', 'rmu', 'rsigma']), extname='N2')
    allpdf = AddPdf(g1, g2)
    bx2 = BinnedChi2(allpdf, data)
    bx2.draw(args=(0, 1, 10000, 3, 1, 10000), parts=True)


@image_comparison('draw_simultaneous.png')
def test_draw_simultaneous():
    np.random.seed(0)
    data = np.random.randn(10000)
    shifted = data + 3.
    g1 = rename(gaussian, ['x', 'lmu', 'sigma'])
    g2 = rename(gaussian, ['x', 'rmu', 'sigma'])
    ulh1 = UnbinnedLH(g1, data)
    ulh2 = UnbinnedLH(g2, shifted)
    sim = SimultaneousFit(ulh1, ulh2)
    sim.draw(args=(0, 1, 3))


@image_comparison('draw_simultaneous_prefix.png')
def test_draw_simultaneous_prefix():
    np.random.seed(0)
    data = np.random.randn(10000)
    shifted = data + 3.
    g1 = rename(gaussian, ['x', 'lmu', 'sigma'])
    g2 = rename(gaussian, ['x', 'rmu', 'sigma'])
    ulh1 = UnbinnedLH(g1, data)
    ulh2 = UnbinnedLH(g2, shifted)
    sim = SimultaneousFit(ulh1, ulh2, prefix=['g1_', 'g2_'])
    minuit = Minuit(sim, g1_lmu=0., g1_sigma=1., g2_rmu=0., g2_sigma=1.,
                    print_level=0)
    minuit.migrad()
    sim.draw(minuit)
