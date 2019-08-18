
import numpy as np
import matplotlib.pyplot as plt
import probfit
import iminuit


np.random.seed(0)
data = np.random.randn(10000) * 4 + 1
hist = plt.hist(data, bins=100, histtype='step');
weights = np.ones(100) * 0.1

h = hist[0]
edges = hist[1]

def gauss_pdf(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) **2 / 2. / sigma ** 2)

binned_likelihood = probfit.BinnedLH(gauss_pdf, data_binned=True, bin_contents=h, bin_edges=edges, use_w2=True, weights=weights)

minuit = iminuit.Minuit(binned_likelihood, sigma=3)
minuit.migrad();

fig = plt.figure()
ax = fig.add_subplot(111)

binned_likelihood.draw(minuit,ax=ax);
fig.savefig('run4.png')

