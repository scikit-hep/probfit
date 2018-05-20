import numpy as np
import matplotlib
matplotlib.use('Agg', warn=False)
from probfit.nputil import mid
from probfit.pdf import crystalball, gaussian, doublecrystalball
from probfit.functor import Normalized
from probfit.toy import gen_toy
from probfit._libstat import compute_chi2
from probfit.nputil import vector_apply
from probfit.costfunc import BinnedLH

    
def test_gen_toy():
    np.random.seed(0)
    bound = (-1, 2)
    ntoy = 100000
    toy = gen_toy(doublecrystalball, ntoy, bound=bound,
                  alpha=1., alpha2=1., n=2.,n2=2., mean=1., sigma=0.3, quiet=False)

    assert len(toy) == ntoy

    htoy, bins = np.histogram(toy, bins=1000, range=bound)

    ncball = Normalized(doublecrystalball, bound)

    f = lambda x: ncball(x, 1., 1., 2., 2., 1., 0.3)

    expected = vector_apply(f, mid(bins)) * ntoy * (bins[1] - bins[0])

    htoy = htoy * 1.0
    err = np.sqrt(expected)

    chi2 = compute_chi2(htoy, expected, err)

    print(chi2, len(bins), chi2 / len(bins))

    assert (0.9 < (chi2 / len(bins)) < 1.1)

