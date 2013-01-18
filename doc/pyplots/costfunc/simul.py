from iminuit import Minuit
from probfit import UnbinnedLH, gaussian, SimultaneousFit, rename
from matplotlib import pyplot as plt
from numpy.random import randn, seed

seed(0)
width = 2.
data1 = randn(1000)*width + 1
data2 = randn(1000)*width + 2

#two gaussian with shared width
pdf1 = rename(gaussian, ('x', 'mu_1', 'sigma'))
pdf2 = rename(gaussian, ('x', 'mu_2', 'sigma'))

lh1 = UnbinnedLH(pdf1, data1)
lh2 = UnbinnedLH(pdf2, data2)

simlh = SimultaneousFit(lh1, lh2)

m = Minuit(simlh, mu_1=1.2, mu_2=2.2, sigma=1.5)

plt.figure(figsize=(8, 3))
plt.subplot(211)
simlh.draw(m)
plt.suptitle('Before')

m.migrad() # fit

plt.figure(figsize=(8, 3))
plt.subplot(212)
simlh.draw(m)
plt.suptitle('After')
