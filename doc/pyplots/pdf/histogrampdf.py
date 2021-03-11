# -*- coding: utf-8 -*-
import numpy as np
from iminuit import Minuit

from probfit import AddPdf, BinnedLH, Extended, gen_toy
from probfit.pdf import HistogramPdf

bound = (0, 10)
np.random.seed(0)
bkg = gen_toy(lambda x: x ** 2, 100000, bound=bound)  # a parabola background
sig = np.random.randn(50000) + 5  # a Gaussian signal
data = np.concatenate([sig, bkg])
# fill histograms with large statistics
hsig, be = np.histogram(sig, bins=40, range=bound)
hbkg, be = np.histogram(bkg, bins=be, range=bound)
# randomize data
data = np.random.permutation(data)
fitdata = data[:1000]

psig = HistogramPdf(hsig, be)
pbkg = HistogramPdf(hbkg, be)
epsig = Extended(psig, extname="N1")
epbkg = Extended(pbkg, extname="N2")
pdf = AddPdf(epbkg, epsig)

blh = BinnedLH(pdf, fitdata, bins=40, bound=bound, extended=True)
m = Minuit(blh, N1=330, N2=670, error_N1=20, error_N2=30)
# m.migrad()
blh.draw(m, parts=True)
