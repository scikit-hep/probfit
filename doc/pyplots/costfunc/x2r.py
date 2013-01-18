from iminuit import Minuit
from probfit import Chi2Regression, linear
from matplotlib import pyplot as plt
from numpy.random import randn, seed
import numpy as np

seed(0)
ndata = 30
x = np.linspace(-10, 10, ndata)
y = 2*x + 5
y += randn(ndata)

x2r = Chi2Regression(linear, x, y, np.array([1.]*ndata))

m = Minuit(x2r, m=1, c=2)

plt.figure(figsize=(8, 3))
plt.subplot(121)
x2r.draw(m)
plt.title('Before')

m.migrad() # fit

plt.subplot(122)
x2r.draw(m)
plt.title('After')
