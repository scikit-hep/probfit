import os
from os.path import dirname, join
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.testing.compare import compare_images
from iminuit import Minuit
from probfit.plotting import draw_pdf, draw_compare_hist
from probfit.pdf import gaussian, linear
from probfit.funcutil import rename
from probfit.functor import Extended, AddPdfNorm, AddPdf
from probfit.costfunc import UnbinnedLH, BinnedLH, BinnedChi2, Chi2Regression, \
                             SimultaneousFit

class image_comparison:
    def __init__(self, baseline):

        baselineimage = join(dirname(__file__), 'baseline', baseline)
        actualimage = join(os.getcwd(), 'actual', baseline)

        self.baseline = baseline
        self.baselineimage = baselineimage
        self.actualimage = actualimage

        try:
            os.makedirs(dirname(actualimage))
        except OSError:
            pass

    def setup(self):
        from matplotlib import rcParams, rcdefaults, use
        use('Agg', warn=False)  # use Agg backend for these tests

        # These settings *must* be hardcoded for running the comparison
        # tests and are not necessarily the default values as specified in
        # rcsetup.py
        rcdefaults()  # Start with all defaults
        rcParams['font.family'] = 'Bitstream Vera Sans'
        rcParams['text.hinting'] = False
        rcParams['text.hinting_factor'] = 8
        rcParams['text.antialiased'] = False


    def test(self):
        # compare_images
        self.setup()
        x = compare_images(self.baselineimage, self.actualimage, 0.007)
        if x is not None:
            print x
            assert x is None


    def __call__(self, f):
        def tmp():
            f()
            plt.savefig(self.actualimage)
            return self.test()
        tmp.__name__ = f.__name__
        return tmp


@image_comparison('draw_pdf.png')
def test_draw_pdf():
    plt.figure()
    f = gaussian
    draw_pdf(f, {'mean':1., 'sigma':2.}, bound=(-10, 10))


@image_comparison('draw_pdf_linear.png')
def test_draw_pdf_linear():
    plt.figure()
    f = linear
    draw_pdf(f, {'m':1., 'c':2.}, bound=(-10, 10))


@image_comparison('draw_compare_hist_gaussian.png')
def test_draw_compare_hist():
    plt.figure()
    np.random.seed(0)
    data = np.random.randn(10000)
    f = gaussian
    draw_compare_hist(f, {'mean':0., 'sigma':1.}, data, normed=True)


@image_comparison('draw_compare_hist_no_norm.png')
def test_draw_compare_hist_no_norm():
    plt.figure()
    np.random.seed(0)
    data = np.random.randn(10000)
    f = Extended(gaussian)
    draw_compare_hist(f, {'mean':0., 'sigma':1., 'N':10000}, data, normed=False)


@image_comparison('draw_ulh.png')
def test_draw_ulh():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    ulh = UnbinnedLH(gaussian, data)
    ulh.draw(args=(0., 1.))

@image_comparison('draw_ulh_extend.png')
def test_draw_ulh_extend():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    ulh = UnbinnedLH(Extended(gaussian), data, extended=True)
    ulh.draw(args=(0., 1., 1000))

@image_comparison('draw_residual_ulh.png')
def test_draw_residual_ulh():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    ulh = UnbinnedLH(gaussian, data)
    ulh.draw_residual(args=(0., 1.))

@image_comparison('draw_residual_ulh_norm.png')
def test_draw_residual_ulh_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    ulh = UnbinnedLH(gaussian, data)
    ulh.draw_residual(args=(0., 1.), norm=True)

@image_comparison('draw_ulh_extend_residual_norm.png')
def test_draw_ulh_extend_residual_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    ulh = UnbinnedLH(Extended(gaussian), data, extended=True)
    ulh.draw_residual(args=(0., 1., 1000), norm=True)

@image_comparison('draw_ulh_with_minuit.png')
def test_draw_ulh_with_minuit():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    ulh = UnbinnedLH(gaussian, data)
    minuit = Minuit(ulh, mean=0, sigma=1)
    ulh.draw(minuit)


@image_comparison('draw_blh.png')
def test_draw_blh():
    np.random.seed(0)
    data = np.random.randn(1000)
    blh = BinnedLH(gaussian, data)
    plt.figure()
    blh.draw(args=(0., 1.))


@image_comparison('draw_blh_extend.png')
def test_draw_blh_extend():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    blh = BinnedLH(Extended(gaussian), data, extended=True)
    blh.draw(args=(0., 1., 1000))

@image_comparison('draw_residual_blh.png')
def test_draw_residual_blh():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    blh = BinnedLH(gaussian, data)
    blh.draw_residual(args=(0., 1.))

@image_comparison('draw_residual_blh_norm.png')
def test_draw_residual_blh_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    blh = BinnedLH(gaussian, data)
    blh.draw_residual(args=(0., 1.), norm=True)

@image_comparison('draw_blh_extend_residual_norm.png')
def test_draw_blh_extend_residual_norm():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    blh = BinnedLH(Extended(gaussian), data, extended=True)
    blh.draw_residual(args=(0., 1., 1000), norm=True)

@image_comparison('draw_bx2.png')
def test_draw_bx2():
    np.random.seed(0)
    data = np.random.randn(1000)
    plt.figure()
    blh = BinnedChi2(Extended(gaussian), data)
    blh.draw(args=(0., 1., 1000))


@image_comparison('draw_x2reg.png')
def test_draw_x2reg():
    np.random.seed(0)
    x = np.linspace(0, 1, 100)
    y = 10.*x + np.random.randn(100)
    err = np.array([1] * 100)
    plt.figure()
    blh = Chi2Regression(linear, x, y, err)
    blh.draw(args=(10., 0.))

@image_comparison('draw_ulh_with_parts.png')
def test_ulh_with_parts():
    np.random.seed(0)
    data = np.random.randn(10000)
    shifted = data + 3.
    data = np.append(data, [shifted])
    print len(data)
    plt.figure()
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
    print len(data)
    plt.figure()
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
    plt.figure()
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
    plt.figure()
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
    plt.figure()
    g1 = rename(gaussian, ['x', 'lmu', 'sigma'])
    g2 = rename(gaussian, ['x', 'rmu', 'sigma'])
    ulh1 = UnbinnedLH(g1, data)
    ulh2 = UnbinnedLH(g2, shifted)
    sim = SimultaneousFit(ulh1, ulh2, prefix=['g1_', 'g2_'])
    minuit = Minuit(sim, g1_lmu=0., g1_sigma=1., g2_rmu=0., g2_sigma=1.,
                    print_level=0)
    minuit.migrad()
    sim.draw(minuit)
