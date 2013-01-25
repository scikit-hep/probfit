from iminuit import Minuit
from probfit import UnbinnedLH, gaussian, Extended
from matplotlib import pyplot as plt
from numpy.random import randn

data = randn(1000)*2 + 1

ulh = UnbinnedLH(gaussian, data)
m = Minuit(ulh, mean=0., sigma=0.5)

plt.figure(figsize=(8, 6))
plt.subplot(221)
ulh.draw(m)
plt.title('Unextended Before')

m.migrad() # fit

plt.subplot(222)
ulh.draw(m)
plt.title('Unextended After')

#Extended

data = randn(2000)*2 + 1
egauss = Extended(gaussian)
ulh = UnbinnedLH(egauss, data, extended=True, extended_bound=(-10.,10.))
m = Minuit(ulh, mean=0., sigma=0.5, N=1800.)

plt.subplot(223)
ulh.draw(m)
plt.title('Extended Before')

m.migrad() # fit

plt.subplot(224)
ulh.draw(m)
plt.title('Extended After')
