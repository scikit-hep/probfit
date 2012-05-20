# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

from dist_fit import *
import numpy as np
import minuit
import UserDict

# <codecell>

#it defines some c function(no python call)
#will wrap this one soon this shows the argument of the function
print describe(crystalball)# this shows all the parameter name
vc = np.vectorize(crystalball)
x = linspace(-1,2,100)
plot(x,vc(x,1,1,1,1))
ylim(ymin=0)
#note that this is not normalized

# <codecell>

#automatic cache normalization of any distribution (for given range)
#you can skip this if your function is properly normalize
ncball = Normalize(crystalball,(-1,2))
print ncball(1.,1.,1.,1.,1.)

# <codecell>

print describe(ncball)
#and you can do toy generation from any distribution as well
#and compare the result for you
#if you are getting weird result from the fit make sure the normalization is the correct range
g = gen_toy(ncball,10000,(-1,2),alpha=1.,n=2.,mean=1.,sigma=1.,quiet=False)

# <codecell>

#yep it works
hist(g,bins=100,histtype='step');
#it's just a normal numpy array
print g

# <codecell>

#now you can fit
#takes argument like one in pyminuit
# give the named parameter to tell wheree to start
# and limit_name = tuple to tell it the range
uml,minu = fit_uml(ncball,g,alpha=1.,n=2.,mean=1.2,sigma=1.,
    limit_n=(1.,10.),limit_sigma=(0.01,3),limit_mean=(0.7,1),quiet=False)
uml.show(minu)
#see it works
#you can access it like a map
print minu.values

# <codecell>

#lets try a function that builtin one is not availble
#this is slow though you can speed this up by using cython but for most purpose this is fast "enough"
peak1 = randn(10000)
peak2 = randn(5000)+10
twopeak = np.append(peak1,peak2)
hist(twopeak,bins=100,histtype='step');

# <codecell>

#you can define you own funciton and use automatic normalization
#I know not all functions have analytic formula for normalization
@normalized_function(-20,20)
def tofit(x,m1,s1,m2,s2,a):
    g1 = gaussian(x,m1,s1)
    g2 = gaussian(x,m2,s2)
    ret = a*g1+(1-a)*g2
    return ret

# <codecell>

#see all good
uml, minu = fit_uml(tofit,twopeak,m1=0.,m2=10.,s1=2.,s2=2.,a=.6)
uml.draw(minu)

# <codecell>

#now what if things doesn't fit
uml, minu = fit_uml(tofit,twopeak,m1=0.,m2=10.,s1=2.,s2=2.,a=3.)
print minu.migrad_ok(),minu.matrix_accurate()

# <codecell>

#chi2 fit is also there
#it will refuse to fit histogram with zero bin though
#it is meaning less to do that anyway

#Remember our ncball we can extend it
#this add N to the end of argument list (N is meaningful only when funciton is normalized)
encball = Extend(ncball)
print encball.func_code.co_varnames[:encball.func_code.co_argcount]

#and fit it with chi^2
uml,minu = fit_binx2(encball,g,alpha=1.,n=2.,mean=1.,sigma=1.,N=7000.,
    limit_n=(1.,10.),limit_sigma=(0.01,3),limit_mean=(0.7,1),quiet=False)
uml.show(minu)

# <codecell>

#you can try to get other information from minuit object
dir(minu)

# <codecell>

#there is also binned poisson
uml,minu = fit_binlh(encball,g,alpha=1.,n=2.,mean=1.,sigma=1.,N=7000.,
    limit_n=(1.,10.),limit_sigma=(0.01,3),limit_mean=(0.7,1),quiet=False,extended=True)
uml.show(minu)

# <codecell>

#or unextended one
uml,minu = fit_binlh(ncball,g,alpha=1.,n=2.,mean=1.,sigma=1.,N=7000.,
    limit_n=(1.,10.),limit_sigma=(0.01,3),limit_mean=(0.7,1),quiet=False,extended=False)
uml.show(minu)

# <codecell>

#guessing initial parameter can be hard so I made these for you
try_uml(tofit,twopeak,m1=0.,m2=10.,s1=2.,s2=2.,a=0.1)

# <codecell>

#take list too so you can try a bunch of parameters at once
#it returns the best one
besttry = try_uml(tofit,twopeak,m1=0.,m2=10.,s1=1.,s2=1.,a=(0.4,0.5,0.7,0.9))
print besttry

# <codecell>

#a nice trick is to use keyword expansion on the return argument
uml,minu = fit_uml(tofit,twopeak,**besttry)
uml.show(minu)

# <codecell>

#showing contour is simple as well
x,y = val_contour(uml,minu,'m1')
plot(x,y)

# <codecell>

#lets try to do chi^2 regression
def f(x,a,b,c):
    return a*x**2+b*x+c
vf = vectorize(f)
x=linspace(-3,3,100)
y = vf(x,1,2,3)+randn(100)#that line + gauassian of mean 0 width 1 to give it some fluctuation
err = np.zeros(100)
err.fill(1.)
errorbar(x,y,err,fmt='.b')
xlim(-3,3)
#now we got y and x

# <codecell>

to_minimize = Chi2Regression(f,x,y,err)
m=minuit.Minuit(to_minimize)
m.migrad()

# <codecell>

print m.values
print m.errors
to_minimize.draw(m)

# <codecell>

#fixing parameter is easy to through minuit
to_minimize = Chi2Regression(f,x,y,err)
m=minuit.Minuit(to_minimize,c=3.,fix_c=True)#just that
m.migrad()
print m.values
print m.errors#yep error on c is the default one
to_minimize.draw(m)

# <codecell>

#there are also nifty arbitary order polynomial functor
p = Polynomial(2)
print describe(p)
print describe(Polynomial(['a','b','c','d']))#name your parameter if you feel like it
vp = vectorize(p)
x = linspace(-10,10,100)
y = vp(x,3,0,2)
plot(x,y);

# <codecell>

x=linspace(-3,3,100)
y=vp(x,1,2,3)+randn(100)#that line + gauassian of mean 0 width 1 to give it some fluctuation
err = np.zeros(100)
err.fill(1.)
errorbar(x,y,err,fmt='.b');

# <codecell>

to_minimize = Chi2Regression(p,x,y,err)
m=minuit.Minuit(to_minimize)
m.migrad()
to_minimize.draw(m)

# <codecell>


# <codecell>


