# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

from root_numpy import *
import minuit
from dist_fit import *

# <codecell>


# <codecell>

ncball = Normalize(crystalball,(0.15,0.25))

# <codecell>

def z(x,y):
    return (x-1)**2+(y-2)**2
mz = minuit.Minuit(z)

# <codecell>

mz.migrad()
mz.values

# <codecell>

data = root2rec('yesbkg-200MeV.root','emc')
data = data[(data.ecal > 0.15) & (data.ecal < 0.23)]

# <codecell>

hist(data.ecal,bins=100,normed=True,histtype='step',label='aaa');
title('my title')
legend();

# <codecell>

import sys
import traceback
uml = UnbinnedML(ncball,data.ecal)

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


uml.func_code.__dict__

# <codecell>

def test():
    m.printMode=1
    m.migrad()

#%timeit test()
test()
print m.values
print m.errors

# <codecell>

h,e,_ = hist(data.ecal,histtype='step',bins=100,normed=True)
w = e[1]-e[0]
ys = [ncball(x,*m.args) for x in mid(e)]
plot(mid(e),ys);

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

# <codecell>

h,e,_ = hist(twopeak,histtype='step',bins=100,normed=True)
ys = [tofit(x,*m2.args) for x in mid(e)]
plot(mid(e),ys,lw=5,alpha=0.5);

# <codecell>


# <codecell>


