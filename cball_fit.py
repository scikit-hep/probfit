# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

from root_numpy import *
import minuit
from dist_fit import *

# <codecell>

data = root2rec('yesbkg-200MeV.root','emc')
data = data[(data.ecal > 0.15) & (data.ecal < 0.23)]
hist(data.ecal,bins=100);

# <codecell>

ncball = Normalize(crystalball,(-1,1))

# <codecell>

#try_uml(ncball,data.ecal,alpha=1,n=(1,5,20),mean=0.2,sigma=0.01)
print ncball(1.,1.,1.,1.,1.)
g = gen_toy(ncball,10000,(-1,2),alpha=1.,n=2.,mean=1.,sigma=1.,quiet=False)

# <codecell>

hist(g,bins=100);

# <codecell>


# <codecell>

hist(data.ecal,bins=100,normed=True,histtype='step',label='aaa');
legend();

# <codecell>

import sys
import traceback
#uml = BinnedChi2(ncball,data.ecal,range=(0.17,0.21),bins=50)
uml = UnBinnedML(ncball,data.ecal)
limit_alpha = (0.5,1.5)
limit_mean = (0.19,0.21)
limit_sigma=(0.005,0.01)
limit_n=(1,20)
alpha = uniform(*limit_alpha)
mean = uniform(*limit_mean)
sigma = uniform(*limit_sigma)
n = uniform(*limit_n)

m = minuit.Minuit(uml,
                  alpha=1.,mean=0.2,n=2.,sigma=0.01,
                  #alpha=alpha,mean=mean,n=n,sigma=sigma,
                  limit_alpha=limit_alpha,limit_mean=limit_mean,
                  limit_sigma=limit_sigma,limit_n=limit_n
                  )

# <codecell>

def test():
    m.printMode=1
    m.migrad()

#%timeit test()
test()
uml.draw(m)
fwhm = fwhm_f(uml.f,(0.16,0.22),m.args)
vertical_highlight(fwhm)

# <codecell>

peak1 = randn(10000)
peak2 = randn(5000)+10
twopeak = np.append(peak1,peak2)
hist(twopeak,bins=100,histtype='step');

# <codecell>

@normalized_function(-20,20)
def tofit(x,m1,s1,m2,s2,a):
    g1 = gaussian(x,m1,s1)
    g2 = gaussian(x,m2,s2)
    ret = a*g1+(1-a)*g2
    return ret

tpuml = UnbinnedML(tofit,twopeak)

# <codecell>

m2 = minuit.Minuit(tpuml,m1=0.,m2=10.,s1=2.,s2=2.,a=0.5)
m2.printMode=1
m2.migrad()
print m2.values
print m2.errors
tpuml.draw(m2)

# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


