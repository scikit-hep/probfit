# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# dist_fit Basic Tutorial

# <codecell>

from probfit import *
import numpy as np
import iminuit
import UserDict
from math import sqrt, exp, pi

# <headingcell level=2>

# Lets start with a simple simple straight line

# <rawcell>

# We can't really call this a fitting package without being able to fit a straight line, right?

# <codecell>

#lets make s straight line with gaussian(mu=0,sigma=1) error
x = linspace(0,10,20) 
y = 3*x+15+ randn(20)
err = np.array([1]*20)
errorbar(x,y,err,fmt='.');

# <codecell>

#lets define our line
#first argument has to be independent variable
#arguments after that are shape parameters
def line(x,m,c): #define it to be parabolic or whatever you like
    return m*x+c
#We can make it faster but for this example this is plentily fast.
#We will talk about speeding things up later(you will need cython)

# <codecell>

describe(line)

# <codecell>

#cost function
chi2 = Chi2Regression(line,x,y,err)

# <codecell>

#Chi^2 regression is just a callable object nothing special about it
describe(chi2)

# <codecell>

#minimize it
#yes it gives you a heads up that you didn't give it initial value
#we can ignore it for now
minimizer = RTMinuit.Minuit(chi2) #see RTMinuit tutorial on how to give initial value/range/error
minimizer.migrad(); #very stable robust minimizer
#you can look at your terminal to see what it is doing

# <codecell>

#lets see our results
print minimizer.values
print minimizer.errors

# <codecell>

#or a pretty printing
minimizer.html_results()
#you can also call display(minimizer.html_results()) if this is in the middle of your code

# <rawcell>

# Parabolic error
# is calculated using the second derivative at the minimum
# This is good in most cases where the uncertainty is symmetric not much correlation
# exists. Migrad usually got this accurately but if you want ot be sure
# call minimizer.hesse() after migrad
# Minos Error
# is obtained by scanning chi^2 and likelihood profile and find the the point
# where chi^2 is increased by 1 or -ln likelihood is increased by 0.5
# This error is generally asymmetric and is correct in all case.
# This is quite computationally expensive if you have many parameter.
# call minimizer.minos('variablename') after migrad to get this error

# <codecell>

#let's visualize our line
chi2.draw(minimizer)
#looks good

# <codecell>

#Sometime we want the error matrix
print 'error matrix:\n', minimizer.error_matrix()
#or correlation matrix
print 'correlation matrix:\n', minimizer.error_matrix(correlation=True)
#or a pretty html
#despite the method named error_matrix it's actually a correlation matrix
display(minimizer.html_error_matrix())

# <headingcell level=2>

# Simple gaussian distribution fit

# <rawcell>

# In high energy physics, we usually want to fit a distribution to a histogram. Let's look at simple gaussian distribution

# <codecell>

#First lets make our data
data = randn(10000)*4+1
#sigma = 4 and mean = 1
hist(data,bins=100,histtype='step');

# <rawcell>

# Here is our PDF/model

# <codecell>

#normalized gaussian
def myPDF(x,mu,sigma):
    return 1/sqrt(2*pi)/sigma*exp(-(x-mu)**2/2./sigma**2)

# <codecell>

#build your cost function here we use binned likelihood
cost = BinnedLH(myPDF,data)

# <codecell>

#Let's fit
minimizer = RTMinuit.Minuit(cost,sigma=3.) #notice here that we give initial value to sigma

#up parameter determine where it determine uncertainty
#1(default) for chi^2 and 0.5 for likelihood
minimizer.set_up(0.5)
minimizer.migrad() #very stable minimization algorithm
#like in all binned fit with long zero tail. It will have to do something about the zero bin
#dist_fit handle them gracefully but will give you a head up

# <codecell>

#let's see the result
print 'Value:', minimizer.values
print 'Error:', minimizer.errors

# <codecell>

#That printout can get out of hand quickly
display(minimizer.html_results())
#and correlation matrix
#will not display well in firefox(tell them to fix writing-mode:)
display(minimizer.html_error_matrix()) 

# <codecell>

#you can see if your fit make any sense too
cost.draw(minimizer)#uncertainty is given by symetric poisson

# <codecell>

#how about making sure the error making sense
draw_contour(cost,minimizer,'mu')

# <codecell>

#2d contour error
#you can notice that it takes sometime to draw
#we will this is because our PDF is defined in Python
#we will show how to speed this up later
draw_contour2d(cost,minimizer,'mu','sigma');

# <headingcell level=2>

# How about Chi^2

# <rawcell>

# Let's explore another popular cost function chi^2

# <codecell>

#we will use the same data as in the previous example
np.random.seed(0)
data = randn(10000)*4+1
#sigma = 4 and mean = 1
hist(data,bins=100,histtype='step');

# <codecell>

#And the same PDF: normalized gaussian
def myPDF(x,mu,sigma):
    return 1/sqrt(2*pi)/sigma*exp(-(x-mu)**2/2./sigma**2)

# <codecell>

#binned chi^2 fit only makes sense for extended fit
extended_pdf = Extend(myPDF)

# <codecell>

#very useful method to look at function signature
describe(extended_pdf) #you can look at what your pdf means

# <codecell>

#Chi^2 distribution fit is really bad for distribution with long tail
#since when bin count=0... poisson error=0 and blows up chi^2
#so give it some range
chi2 = BinnedChi2(extended_pdf,data,range=(-7,10))
minimizer = RTMinuit.Minuit(chi2,sigma=1)
minimizer.migrad()

# <codecell>

chi2.draw(minimizer)

# <codecell>

#and usual deal
display(minimizer.html_results())
display(minimizer.html_error_matrix()) 

# <headingcell level=2>

# Unbinned Likelihood and How to speed things up

# <rawcell>

# Unbinned likelihood is computationally very very expensive.
# It's now a good time that we talk about how to speed things up with cython

# <codecell>

#same data
np.random.seed(0)
data = randn(10000)*4+1
#sigma = 4 and mean = 1
hist(data,bins=100,histtype='step');

# <codecell>

#We want to speed things up with cython
#load cythonmagic
%load_ext cythonmagic

# <codecell>

%%cython
cimport cython
from libc.math cimport exp,M_PI,sqrt
#same gaussian distribution but now written in cython
@cython.binding(True)#IMPORTANT:this tells cython to dump function signature too
def cython_PDF(double x,double mu,double sigma):
    #these are c add multiply etc not python so it's fast
    return 1/sqrt(2*M_PI)/sigma*exp(-(x-mu)**2/2./sigma**2)

# <codecell>

#cost function 
ublh = UnbinnedML(cython_PDF,data)
minimizer = RTMinuit.Minuit(ublh,sigma=2.)
minimizer.set_up(0.5)#remember this is likelihood
minimizer.migrad()#yes amazingly fast
ublh.show(minimizer)
display(minimizer.html_results())
display(minimizer.html_error_matrix()) 

# <codecell>

#remember how slow it was?
#now it's super fast(and it's even unbinned likelihood)
draw_contour(ublh,minimizer,'mu')

# <codecell>

#but you really don't have to write your own gaussian 
#there are tons of builtin function written in cython for you
print describe(gaussian)
print type(gaussian)

# <codecell>

ublh = UnbinnedML(gaussian,data)
minimizer = RTMinuit.Minuit(ublh,sigma=2.)
minimizer.set_up(0.5)#remember this is likelihood
minimizer.migrad()#yes amazingly fast
ublh.show(minimizer)
display(minimizer.html_results())
display(minimizer.html_error_matrix()) 

# <headingcell level=2>

# But... We can't normalize everything analytically and how to generate toy sample from PDF

# <rawcell>

# When fitting distribution to a PDF. One of the common problem that we run into is normalization.
# Not all function is analytically integrable on the range of our interest.
# Let's look at crystal ball function

# <codecell>

#lets first generate a crystal ball sample
#dist_fit has builtin toy generation capability
#lets introduce crystal ball function
#http://en.wikipedia.org/wiki/Crystal_Ball_function
#it's simply gaussian with power law tail
#normally found in energy deposited in crystals
#impossible to normalize analytically
#and normalization will depend on shape parameters
describe(crystalball)

# <codecell>

np.random.seed(0)
bound = (-1,2)
data = gen_toy(crystalball,10000,bound=bound,alpha=1.,n=2.,mean=1.,sigma=0.3,quiet=False)
#quiet = False tells it to plot out original function
#toy histogram and poisson error from both orignal distribution and toy

# <codecell>

#To fit this we need to tell normalized our crystal ball PDF
#this is done with trapezoid rule with simple cache mechanism
#can be done by Normalize functor
ncball = Normalize(crystalball,bound)
#this can also bedone with declerator
#@normalized_function(bound)
#def myPDF(x,blah):
#    return blah
print 'unnorm:', crystalball(1.0,1,2,1,0.3)
print '  norm:', ncball(1.0,1,2,1,0.3)

# <codecell>

#it has the same signature as the crystalball
describe(ncball)

# <codecell>

#now we can fit as usual
ublh = UnbinnedML(ncball,data)
minimizer = RTMinuit.Minuit(ublh,
    alpha=1.,n=2.1,mean=1.2,sigma=0.3)
minimizer.set_up(0.5)#remember this is likelihood
minimizer.migrad()#yes amazingly fast Normalize is written in cython
ublh.show(minimizer)
#crystalball function is nortorious for its sensitivity on n parameter
#dist_fit give you a heads up where it might have float overflow

# <rawcell>

# Bonus: one thing we normally ask about asymmetric peaking distribution like this is full-width-half-max
# http://en.wikipedia.org/wiki/Full_width_at_half_maximum

# <codecell>

lw,rw = fwhm_f(ncball,(-1,2),minimizer.args)

# <codecell>

ublh.draw(minimizer)
axvspan(lw,rw,alpha=0.2,color='green');

# <headingcell level=2>

# What if things went wrong

# <codecell>

#crystalball is nortoriously sensitive to initial parameter
#now it is a good time to show what happen when things...doesn't fit
ublh = UnbinnedML(ncball,data)
minimizer = RTMinuit.Minuit(ublh)#NO initial value
minimizer.set_up(0.5)#remember this is likelihood
minimizer.migrad()#yes amazingly fast
#Remember there is a heads up

# <codecell>

ublh.show(minimizer)#it bounds to fails

# <codecell>

minimizer.migrad_ok(), minimizer.matrix_accurate()
#checking these two method give you a good sign

# <codecell>

#fix it by giving it initial value/error/limit or fixing parameter see RTMinuit Tutorial
#now we can fit as usual
ublh = UnbinnedML(ncball,data)
minimizer = RTMinuit.Minuit(ublh,
    alpha=1.,n=2.1,mean=1.2,sigma=0.3)
minimizer.set_up(0.5)#remember this is likelihood
minimizer.migrad()#yes amazingly fast. Normalize is written in cython
ublh.show(minimizer)

# <headingcell level=2>

# How do we guess initial value?

# <rawcell>

# This is a hard question but visualization can helps us

# <codecell>

besttry = try_uml(ncball,data,alpha=1.,n=2.1,mean=1.2,sigma=0.3)

# <codecell>

#or you can try multiple
#too many will just confuse you
besttry = try_uml(ncball,data,alpha=1.,n=2.1,mean=[1.2,1.1],sigma=[0.3,0.5])
print besttry #and you can find which one give you minimal unbinned likelihood

# <headingcell level=2>

# Extended Fit: 2 Gaussian with Polynomial Background

# <codecell>

peak1 = randn(3000)*0.2
peak2 = randn(5000)*0.1+4
bg = gen_toy(lambda x : (x+2)**2, 20000,(-2,5))
all_data = np.concatenate([peak1,peak2,bg])
hist((peak1,peak2,bg,all_data),bins=200,histtype='step',range=(-2,5));

# <codecell>

%%cython
cimport cython
from dist_fit import Normalize, gaussian

@cython.binding(True)
def poly(double x,double a,double b, double c):
    return a*x*x+b*x+c

#remember linear function is not normalizable for -inf ... +inf
nlin = Normalize(poly,(-1,5))

#our extended PDF for 3 types of signal
@cython.binding(True)
def myPDF(double x, 
    double a, double b, double c, double nbg,
    double mu1, double sigma1, double nsig1,
    double mu2, double sigma2, double nsig2):

    cdef double NBG = nbg*nlin(x,a,b,c)
    cdef double NSIG1 = nsig1*gaussian(x,mu1,sigma1) 
    cdef double NSIG2 = nsig2*gaussian(x,mu2,sigma2)
    return NBG + NSIG1 + NSIG2

# <codecell>

print describe(myPDF)

# <codecell>

#lets use what we just learned
#for complicated function good initial value(and initial step) is crucial
#if it doesn't converge try play with initial value and initial stepsize(error_xxx)
besttry = try_binlh(myPDF,all_data, 
    a=1.,b=2.,c=4.,nbg=20000.,
    mu1=0.1,sigma1=0.2,nsig1=3000.,
    mu2=3.9,sigma2=0.1,nsig2=5000.,extended=True, bins=300, bound=(-1,5) )
print besttry

# <codecell>

binlh = BinnedLH(myPDF,all_data, bins=200, extended=True, range=(-1,5))
#too lazy to type initial values from what we try?
#use keyword expansion **besttry
#need to put in initial step size for mu1 and mu2 
#with error_mu* otherwise it won't converge(try it yourself)
minimizer = RTMinuit.Minuit(binlh, error_mu1=0.1, error_mu2=0.1, **besttry)

# <codecell>

minimizer.migrad()

# <codecell>

binlh.show(minimizer)

# <codecell>

display(minimizer.html_results())
minimizer.html_error_matrix()

# <headingcell level=2>

# Advance: Custom cost function and Simultaneous Fit

# <rawcell>

# Sometimes, what we want to fit is the sum of likelihood /chi^2 of two PDF sharing some parameters.
# dist_fit doesn't provied a built_in facility to do this but it can be built easily.
# 
# In this example, we will fit two gaussian distribution where we know that the width are the same
# but the peak is at different places.

# <codecell>

np.random.seed(0)
#Lets make two gaussian
data1 = randn(10000)+3
data2 = randn(10000)-2
hist([data1,data2],bins=100,histtype='step',label=['data1','data2']);

# <codecell>

#remember this is nothing special about builtin cost function
#except some utility function like draw and show
ulh1 = UnbinnedML(gaussian,data1)
ulh2 = UnbinnedML(gaussian,data2)
print describe(ulh1)
print describe(ulh2)

# <codecell>

#you can also use cython to do this
class CustomCost:
    def __init__(self,pdf1,data1,pdf2,data2):
        self.ulh1 = UnbinnedML(pdf1,data1)
        self.ulh2 = UnbinnedML(pdf2,data2)
    #this is the important part you need __call__ to calculate your cost
    #in our case it's sum of likelihood with sigma
    def __call__(self,mu1,mu2,sigma):
        return self.ulh1(mu1,sigma)+self.ulh2(mu2,sigma)

# <codecell>

simul_lh = CustomCost(gaussian,data1,gaussian,data2)

# <codecell>

minimizer = RTMinuit.Minuit(simul_lh,sigma=0.5)
minimizer.set_up(0.5)#remember it's likelihood
minimizer.migrad()

# <codecell>

display(minimizer.html_results())
display(minimizer.html_error_matrix())
results = minimizer.values

# <codecell>

draw_compare_hist(gaussian,[results['mu1'],results['sigma']],data1,normed=True);
draw_compare_hist(gaussian,[results['mu2'],results['sigma']],data2,normed=True);

