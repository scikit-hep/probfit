from iminuit import Minuit
from probfit import BinnedLH, gaussian, Extended
from matplotlib import pyplot as plt
from numpy.random import randn

data = randn(1000)*2 + 1

#Unextended

blh = BinnedLH(gaussian, data)
#if you wonder what it loos like call desceibe(blh)
m = Minuit(blh, mean=0., sigma=0.5)

plt.figure(figsize=(8, 6))
plt.subplot(221)
blh.draw(m)
plt.title('Unextended Before')

m.migrad() # fit

plt.subplot(222)
blh.draw(m)
plt.title('Unextended After')

#Extended

ext_gauss = Extended(gaussian)

blh = BinnedLH(ext_gauss, data, extended=True)
m = Minuit(blh, mean=0., sigma=0.5, N=900.)

plt.subplot(223)
blh.draw(m)
plt.title('Extended Before')

m.migrad()

plt.subplot(224)
blh.draw(m)
plt.title('Extended After')
