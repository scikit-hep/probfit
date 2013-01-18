from iminuit import Minuit
from probfit import BinnedChi2, Extended, gaussian
from matplotlib import pyplot as plt
from numpy.random import randn, seed

seed(0)
data = randn(1000)*2 + 1

ext_gauss = Extended(gaussian)
ulh = BinnedChi2(ext_gauss, data)

m = Minuit(ulh, mean=0., sigma=0.5, N=800)

plt.figure(figsize=(8, 3))
plt.subplot(121)
ulh.draw(m)
plt.title('Before')

m.migrad() # fit

plt.subplot(122)
ulh.draw(m)
plt.title('After')
