from iminuit import Minuit
from probfit import AddPdf, BinnedLH, gaussian, Extended, rename
from matplotlib import pyplot as plt
from numpy.random import randn
import numpy as np

peak1 = randn(1000)*0.5 + 1.
peak2 = randn(500)*0.5 + 0.
#two peaks data with shared width
data = np.concatenate([peak1, peak2])

#Share the width
#If you use Normalized here. Do not reuse the object.
#It will be really slow due to cache miss. Read Normalized doc for more info.
pdf1 = rename(gaussian, ('x', 'm_1', 'sigma'))
pdf2 = rename(gaussian, ('x', 'm_2', 'sigma'))

ext_pdf1 = Extended(pdf1, extname='N_1')
ext_pdf2 = Extended(pdf2, extname='N_2')

compdf = AddPdf(ext_pdf1, ext_pdf2) # merge by name (merge sigma)

ulh = BinnedLH(compdf, data, extended=True)
m = Minuit(ulh, m_1=0.1, m_2=-0.1, sigma=0.1, N_1=900, N_2=480)

plt.figure(figsize=(8, 3))
plt.subplot(121)
ulh.draw(m, parts=True)
plt.title('Before')

m.migrad() # fit

plt.subplot(122)
ulh.draw(m, parts=True)
plt.title('After')
