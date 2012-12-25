import matplotlib
matplotlib.use('template')
from nose.tools import *
import numpy.random as npr
import numpy as np
from probfit.nputil import mid
from probfit.cdist_func import crystalball
from probfit.functor import Normalized
from probfit.toy import gen_toy
from probfit.util import describe
from probfit._libstat import compute_chi2
from probfit.nputil import vector_apply

def test_gen_toy():
    npr.seed(0)
    bound = (-1,2)
    ntoy = 100000
    toy = gen_toy( crystalball,ntoy, bound=bound,
        alpha=1., n=2., mean=1., sigma=0.3, quiet=False)

    assert_equal(len(toy), ntoy)

    htoy, bins = np.histogram(toy, bins=1000, range=bound)

    ncball = Normalized(crystalball,bound)

    f = lambda x: ncball(x, 1., 2., 1., 0.3)

    expected = vector_apply(f, mid(bins))*ntoy*(bins[1]-bins[0])
    #print htoy[:100]
    #print expected[:100]

    htoy = htoy*1.0
    err = np.sqrt(expected)

    chi2 = compute_chi2(htoy, expected, err)

    print chi2, len(bins), chi2/len(bins)

    assert(0.9<(chi2/len(bins))<1.1)
