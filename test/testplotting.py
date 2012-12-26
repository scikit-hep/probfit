from matplotlib.testing.decorators import image_comparison
from probfit.plotting import *
from probfit.pdf import gaussian, linear
from probfit.functor import Extended
from matplotlib.testing.compare import compare_images
from matplotlib import pyplot as plt
from os.path import dirname, join
import numpy.random as npr
import numpy as np
import os
import sys
class image_comparison:
    def __init__(self, baseline):

        baselineimage = join(dirname(__file__),'baseline',baseline)
        actualimage = join(os.getcwd(),'actual',baseline)

        self.baseline = baseline
        self.baselineimage = baselineimage
        self.actualimage = actualimage

        try:
            os.makedirs(dirname(actualimage))
        except OSError as e:
            pass

    def test(self):
        #compare_images
        x = compare_images(self.baselineimage, self.actualimage, 0.005)
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
def test_plot():
    plt.figure()
    f = gaussian
    draw_pdf(f, {'mean':1.,'sigma':2.}, bound=(-10,10))

@image_comparison('draw_pdf_linear.png')
def test_plot2():
    plt.figure()
    f = linear
    draw_pdf(f, {'m':1.,'c':2.}, bound=(-10,10))

@image_comparison('draw_compare_hist_gaussian.png')
def test_draw_compare_hist():
    plt.figure()
    npr.seed(0)
    data = npr.randn(10000)
    edges = np.linspace(-5,5,100)
    f = gaussian
    draw_compare_hist(f, {'mean':0., 'sigma':1.}, data, normed=True)

@image_comparison('draw_compare_hist_no_norm.png')
def test_draw_compare_hist_no_norm():
    plt.figure()
    npr.seed(0)
    data = npr.randn(10000)
    edges = np.linspace(-5,5,100)
    f = Extended(gaussian)
    draw_compare_hist(f, {'mean':0., 'sigma':1., 'N':10000}, data, normed=False)
