from iminuit import Minuit
from probfit import UnbinnedLH, gaussian
from matplotlib import pyplot as plt
from numpy.random import randn

data = randn(1000)*2 + 1

ulh = UnbinnedLH(gaussian, data)
m = Minuit(ulh, mean=0., sigma=0.5)

plt.figure(figsize=(8, 3))
plt.subplot(121)
ulh.draw(m)
plt.title('Before')

m.migrad() # fit

plt.subplot(122)
ulh.draw(m)
plt.title('After')
