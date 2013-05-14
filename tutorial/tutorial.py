## probfit Basic Tutorial

# [probfit](http://iminuit.github.io/probfit/) is a modeling / fitting package to be used together with [iminuit](http://iminuit.github.com/iminuit/).
# 
# This tutorial is a fast-paced introduction to the probfit features:
# 
# * built-in common models: polynomial, gaussian, ...
# * build-in common fit statistics: chi^2, binned and unbinned likelihood
# * tools to get your fits to converge and check the results: try_uml, draw, draw_residuals, ...
# * tools to help you implement your own models and fit statistics: Normalize, Extended, integrate_1d, ...
# 
# Please start this notebook with the ``ipython --pylab=inline`` option to get inline plots.

# In[1]:
# We assume you have executed this cell in all the following examples
import numpy as np
import matplotlib.pyplot as plt
import iminuit
import probfit

# In your own code you can explicitly import what you need to save
# typing in interactive sessions, e.g.
# 
#     from iminuit import Minuit, describe
#     from probfit import gaussian, BinnedLH
# 
# We don't do this here, we only import `iminuit` and `probfit` into our
# namespace so that it is clear to you which functions and classes come
# from which package while reading the code below.

# ## Chi^2 straight line fit
# 
# We can't really call this a fitting package without being able to fit a straight line, right?

# In[2]:
# Let's make a straight line with gaussian(mu=0, sigma=1) noise
np.random.seed(0)
x = np.linspace(0, 10, 20) 
y = 3 * x + 15 + np.random.randn(len(x))
err = np.ones(len(x))
plt.errorbar(x, y, err, fmt='.');

# Out[2]:
# image file: tutorial_files/tutorial_fig_00.png

# In[3]:
# Let's define our line.
# First argument has to be the independent variable,
# arguments after that are shape parameters.
def line(x, m, c): # define it to be parabolic or whatever you like
    return m * x + c

# In[4]:
iminuit.describe(line)

# Out[4]:
#     ['x', 'm', 'c']


# In[5]:
# Define a chi^2 cost function
chi2 = probfit.Chi2Regression(line, x, y, err)

# In[6]:
# Chi2Regression is just a callable object; nothing special about it
iminuit.describe(chi2)

# Out[6]:
#     ['m', 'c']


# In[7]:
# minimize it
# yes, it gives you a heads up that you didn't give it initial value
# we can ignore it for now
minuit = iminuit.Minuit(chi2) # see iminuit tutorial on how to give initial value/range/error
minuit.migrad(); # MIGRAD is a very stable robust minimization method
# you can look at your terminal to see what it is doing;

# Out[7]:
#     -c:4: InitialParamWarning: Parameter m does not have initial value. Assume 0.
#     -c:4: InitialParamWarning: Parameter m is floating but does not have initial step size. Assume 1.
#     -c:4: InitialParamWarning: Parameter c does not have initial value. Assume 0.
#     -c:4: InitialParamWarning: Parameter c is floating but does not have initial step size. Assume 1.
# 
# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 12.0738531135</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 36</td>
#                 <td title="Number of call in last migrad">NCALLS = 36</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 1.10886029888e-21</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 1e-05</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 1.0</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#TJjdDPQkvo').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>m</td>
#                 <td>2.886277e+00</td>
#                 <td>7.367884e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>c</td>
#                 <td>1.613795e+01</td>
#                 <td>4.309458e-01</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="TJjdDPQkvo" style="display:none;">
#             <textarea rows="10" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & m & 2.886e+00 & 7.368e-02 &  &  &  &  & \\
# \hline
# 2 & c & 1.614e+01 & 4.309e-01 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# In[8]:
# The output above is a pretty-printed summary of the fit results from
# minuit.print_fmin()
# which was automatically called by iminuit.Minuit.migrad() after running MIGRAD.

# Let's see our results as Python dictionaries ...
print(minuit.values)
print(minuit.errors)

# Out[8]:
#     {'c': 16.137947520534624, 'm': 2.8862774144823855}
#     {'c': 0.4309458211385722, 'm': 0.07367884284273937}
# 
# #### Parabolic error
# is calculated using the second derivative at the minimum
# This is good in most cases where the uncertainty is symmetric not much correlation
# exists. Migrad usually got this accurately but if you want ot be sure
# call `minuit.hesse()` after calling `minuit.migrad()`.
# 
# #### Minos Error
# is obtained by scanning the chi^2 or likelihood profile and find the point
# where chi^2 is increased by `minuit.errordef`. Note that in the Minuit documentation
# and output `errordef` is often called `up` ... it's the same thing.
# 
# #### What `errordef` should I use?
# 
# As explained in the Minuit documentation you should use:
# 
# * `errordef = 1` for chi^2 fits
# * `errordef = 0.5` for likelihood fits
# 
# `errordef=1` is the default, so you only have to set it to `errordef=0.5`
# if you are defining a likelihood cost function (if you don't your HESSE and MINOS errors will be incorrect).
# `probfit` helps you by defining a `default_errordef()` attribute on the
# cost function classes, which is automatically detected by the `Minuit` constructor
# and can be used to set `Minuit.errordef` correctly, so that users can't forget.
# Classes used in this tutorial:
# 
# * `probfit.Chi2Regression.get_errordef()` and `probfit.BinnedChi2.get_errordef()` return 1.
# * `probfit.BinnedLH.get_errordef()` and `probfit.UnbinnedLH.get_errordef()` return 0.5.

# In[9]:
# Let's visualize our line
chi2.draw(minuit)
# looks good;

# Out[9]:
# image file: tutorial_files/tutorial_fig_01.png

# In[10]:
# Sometimes we want the error matrix (a.k.a. covariance matrix)
print('error matrix:')
print(minuit.matrix())
# or the correlation matrix
print('correlation matrix:')
print(minuit.matrix(correlation=True))
# or a pretty html representation
# Note that `print_matrix()` shows the correlation matrix, not the error matrix
minuit.print_matrix()

# Out[10]:
#     error matrix:
#     ((0.005428571882645087, -0.027142859751431703), (-0.027142859751431703, 0.18571430075679826))
#     correlation matrix:
#     ((1.0, -0.8548504260481388), (-0.8548504260481388, 1.0))
# 
# 
#             <table>
#                 <tr>
#                     <td><a onclick="$('#OVlshZXilM').toggle()" href="#">+</a></td>
#         
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             m
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             c
#             </div>
#             </div>
#             </td>
#             
#                 </tr>
#                 
#             <tr>
#                 <td>m</td>
#             
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(242,137,127)">
#                 -0.85
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>c</td>
#             
#                 <td style="background-color:rgb(242,137,127)">
#                 -0.85
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#             </tr>
#             </table>
# 
#             <pre id="OVlshZXilM" style="display:none;">
#             <textarea rows="13" cols="50" onclick="this.select()" readonly>%\usepackage[table]{xcolor} % include this for color
# %\usepackage{rotating} % include this for rotate header
# %\documentclass[xcolor=table]{beamer} % for beamer
# \begin{tabular}{|c|c|c|}
# \hline
# \rotatebox{90}{} & \rotatebox{90}{m} & \rotatebox{90}{c}\\
# \hline
# m & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{242,137,127} -0.85\\
# \hline
# c & \cellcolor[RGB]{242,137,127} -0.85 & \cellcolor[RGB]{255,117,117} 1.00\\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# ## Binned Poisson likelihood fit of a Gaussian distribution
# In high energy physics, we usually want to fit a distribution to a histogram. Let's look at simple Gaussian distribution.

# In[11]:
# First let's make some example data
np.random.seed(0)
data = np.random.randn(10000) * 4 + 1
# sigma = 4 and mean = 1
plt.hist(data, bins=100, histtype='step');

# Out[11]:
# image file: tutorial_files/tutorial_fig_02.png

# In[12]:
# Define your PDF / model
def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2. / sigma ** 2)

# In[13]:
# Build your cost function
# Here we use binned likelihood
binned_likelihood = probfit.BinnedLH(gauss_pdf, data)

# In[14]:
# Create the minuit
# and give an initial value for the sigma parameter
minuit = iminuit.Minuit(binned_likelihood, sigma=3)
# Remember: minuit.errordef is automatically set to 0.5
# as required for likelihood fits (this was explained above)
binned_likelihood.draw(minuit);

# Out[14]:
#     -c:3: InitialParamWarning: Parameter mu does not have initial value. Assume 0.
#     -c:3: InitialParamWarning: Parameter mu is floating but does not have initial step size. Assume 1.
#     -c:3: InitialParamWarning: Parameter sigma is floating but does not have initial step size. Assume 1.
# 
# image file: tutorial_files/tutorial_fig_03.png

# In[15]:
minuit.migrad()
# Like in all binned fit with long zero tail. It will have to do something about the zero bin
# probfit.BinnedLH does handle them gracefully but will give you a warning;

# Out[15]:
#     -c:1: LogWarning: x is really small return 0
# 
# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 20.9368166553</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 46</td>
#                 <td title="Number of call in last migrad">NCALLS = 46</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 1.45381812456e-06</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 5e-06</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 0.5</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#XlneHvbAXy').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>mu</td>
#                 <td>9.258754e-01</td>
#                 <td>3.962599e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>sigma</td>
#                 <td>3.952381e+00</td>
#                 <td>2.826741e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="XlneHvbAXy" style="display:none;">
#             <textarea rows="10" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & $\mu$ & 9.259e-01 & 3.963e-02 &  &  &  &  & \\
# \hline
# 2 & $\sigma$ & 3.952e+00 & 2.827e-02 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# In[16]:
# Visually check if the fit succeeded by plotting the model over the data
binned_likelihood.draw(minuit) # uncertainty is given by symmetric Poisson;

# Out[16]:
# image file: tutorial_files/tutorial_fig_04.png

# In[17]:
# Let's see the result
print('Value: {}'.format(minuit.values))
print('Error: {}'.format(minuit.errors))

# Out[17]:
#     Value: {'mu': 0.9258754454758255, 'sigma': 3.9523813236078955}
#     Error: {'mu': 0.039625990354239755, 'sigma': 0.028267407263212106}
# 
# In[18]:
# That printout can get out of hand quickly
minuit.print_fmin()
# Also print the correlation matrix
minuit.print_matrix()

# Out[18]:
# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 20.9368166553</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 46</td>
#                 <td title="Number of call in last migrad">NCALLS = 46</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 1.45381812456e-06</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 5e-06</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 0.5</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#bpbpIknPGG').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>mu</td>
#                 <td>9.258754e-01</td>
#                 <td>3.962599e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>sigma</td>
#                 <td>3.952381e+00</td>
#                 <td>2.826741e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="bpbpIknPGG" style="display:none;">
#             <textarea rows="10" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & $\mu$ & 9.259e-01 & 3.963e-02 &  &  &  &  & \\
# \hline
# 2 & $\sigma$ & 3.952e+00 & 2.827e-02 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# 
#             <table>
#                 <tr>
#                     <td><a onclick="$('#KEkawGmLUq').toggle()" href="#">+</a></td>
#         
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             mu
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             sigma
#             </div>
#             </div>
#             </td>
#             
#                 </tr>
#                 
#             <tr>
#                 <td>mu</td>
#             
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 -0.00
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>sigma</td>
#             
#                 <td style="background-color:rgb(163,254,186)">
#                 -0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#             </tr>
#             </table>
# 
#             <pre id="KEkawGmLUq" style="display:none;">
#             <textarea rows="13" cols="50" onclick="this.select()" readonly>%\usepackage[table]{xcolor} % include this for color
# %\usepackage{rotating} % include this for rotate header
# %\documentclass[xcolor=table]{beamer} % for beamer
# \begin{tabular}{|c|c|c|}
# \hline
# \rotatebox{90}{} & \rotatebox{90}{$\mu$} & \rotatebox{90}{$\sigma$}\\
# \hline
# $\mu$ & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{163,254,186} -0.00\\
# \hline
# $\sigma$ & \cellcolor[RGB]{163,254,186} -0.00 & \cellcolor[RGB]{255,117,117} 1.00\\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# In[19]:
# Looking at a likelihood profile is a good method
# to check that the reported errors make sense
minuit.draw_mnprofile('mu');

# Out[19]:
#     -c:3: LogWarning: x is really small return 0
# 
# image file: tutorial_files/tutorial_fig_05.png

# In[20]:
# Plot a 2d contour error
# You can notice that it takes some time to draw
# We will this is because our PDF is defined in Python
# We will show how to speed this up later
minuit.draw_mncontour('mu', 'sigma');

# Out[20]:
#     /Users/deil/Library/Python/2.7/lib/python/site-packages/iminuit/_plotting.py:85: LogWarning: x is really small return 0
#       sigma=this_sig)
# 
# image file: tutorial_files/tutorial_fig_06.png

# ## Chi^2 fit of a Gaussian distribution
# 
# Let's explore another popular cost function chi^2.
# Chi^2 is bad when you have bin with 0.
# ROOT just ignore.
# ROOFIT does something I don't remember.
# But it's best to avoid using chi^2 when you have bin with 0 count.

# In[21]:
# We will use the same data as in the previous example
np.random.seed(0)
data = np.random.randn(10000) * 4 + 1
# sigma = 4 and mean = 1
plt.hist(data, bins=100, histtype='step');

# Out[21]:
# image file: tutorial_files/tutorial_fig_07.png

# In[22]:
# We will use the same PDF as in the previous example
def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) **2 / 2. / sigma ** 2)

# In[23]:
# Binned chi^2 fit only makes sense (for now) for extended PDFs
# probfit.Extended adds a norm parameter with name 'N'
extended_gauss_pdf = probfit.Extended(gauss_pdf)

# In[24]:
# Describe the function signature
iminuit.describe(extended_gauss_pdf)

# Out[24]:
#     ['x', 'mu', 'sigma', 'N']


# In[25]:
# Chi^2 distribution fit is really bad for distribution with long tail
# since when bin count=0... poisson error=0 and blows up chi^2
# so give it some range
chi2 = probfit.BinnedChi2(extended_gauss_pdf, data, bound=(-7,10))
# This time we use the pedantic=False option to tell Minuit
# that we don't want warnings about parameters without initial
# value or step size.
# And print_level=0 means that no output is generated
minuit = iminuit.Minuit(chi2, sigma=1, pedantic=False, print_level=0)
minuit.migrad();

# In[26]:
# Now let's look at the results
minuit.print_fmin()
minuit.print_matrix()
chi2.draw(minuit);

# Out[26]:
# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 36.5025994291</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 247</td>
#                 <td title="Number of call in last migrad">NCALLS = 247</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 3.00607928198e-08</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 1e-05</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 1.0</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#NAJmXiVDsZ').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>mu</td>
#                 <td>9.064759e-01</td>
#                 <td>4.482536e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>sigma</td>
#                 <td>3.959036e+00</td>
#                 <td>4.112828e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>3</td>
#                 <td>N</td>
#                 <td>9.969691e+03</td>
#                 <td>1.033724e+02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="NAJmXiVDsZ" style="display:none;">
#             <textarea rows="12" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & $\mu$ & 9.065e-01 & 4.483e-02 &  &  &  &  & \\
# \hline
# 2 & $\sigma$ & 3.959e+00 & 4.113e-02 &  &  &  &  & \\
# \hline
# 3 & N & 9.970e+03 & 1.034e+02 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# 
#             <table>
#                 <tr>
#                     <td><a onclick="$('#wtxFcCXdZo').toggle()" href="#">+</a></td>
#         
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             mu
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             sigma
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             N
#             </div>
#             </div>
#             </td>
#             
#                 </tr>
#                 
#             <tr>
#                 <td>mu</td>
#             
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(172,240,179)">
#                 -0.10
#                 </td>
#                 
#                 <td style="background-color:rgb(167,247,183)">
#                 -0.05
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>sigma</td>
#             
#                 <td style="background-color:rgb(172,240,179)">
#                 -0.10
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(180,229,173)">
#                 0.18
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>N</td>
#             
#                 <td style="background-color:rgb(167,247,183)">
#                 -0.05
#                 </td>
#                 
#                 <td style="background-color:rgb(180,229,173)">
#                 0.18
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#             </tr>
#             </table>
# 
#             <pre id="wtxFcCXdZo" style="display:none;">
#             <textarea rows="15" cols="50" onclick="this.select()" readonly>%\usepackage[table]{xcolor} % include this for color
# %\usepackage{rotating} % include this for rotate header
# %\documentclass[xcolor=table]{beamer} % for beamer
# \begin{tabular}{|c|c|c|c|}
# \hline
# \rotatebox{90}{} & \rotatebox{90}{$\mu$} & \rotatebox{90}{$\sigma$} & \rotatebox{90}{N}\\
# \hline
# $\mu$ & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{172,240,179} -0.10 & \cellcolor[RGB]{167,247,183} -0.05\\
# \hline
# $\sigma$ & \cellcolor[RGB]{172,240,179} -0.10 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{180,229,173} 0.18\\
# \hline
# N & \cellcolor[RGB]{167,247,183} -0.05 & \cellcolor[RGB]{180,229,173} 0.18 & \cellcolor[RGB]{255,117,117} 1.00\\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# image file: tutorial_files/tutorial_fig_08.png

# ## Fast unbinned likelihood fit Cython
# 
# Unbinned likelihood is computationally very very expensive if you have a lot of data.
# It's now a good time that we talk about how to speed things up with [Cython](http://cython.org).

# In[27]:
# We will use the same data as in the previous example
np.random.seed(0)
data = np.random.randn(10000) * 4 + 1
# sigma = 4 and mean = 1
plt.hist(data, bins=100, histtype='step');

# Out[27]:
# image file: tutorial_files/tutorial_fig_09.png

# In[28]:
# We want to speed things up with Cython
%load_ext cythonmagic

# In[29]:
%%cython
# Same gaussian distribution but now written in Cython
# The %%cython IPython does the following:
# * Call Cython to generate C code for a Python C extension.
# * Compile it into a Python C extension (a shared library)
# * Load it into the current namespace
# If you don't understand these things, don't worry, it basically means:
# * Get full-metal speed easily
cimport cython
from libc.math cimport exp, M_PI, sqrt
@cython.binding(True) # IMPORTANT: this tells Cython to dump the function signature
def gauss_pdf_cython(double x, double mu, double sigma):
    return 1 / sqrt(2 * M_PI) / sigma * exp(-(x - mu) ** 2 / 2. / sigma ** 2)

# In[30]:
# Define the unbinned likelihood cost function 
unbinned_likelihood = probfit.UnbinnedLH(gauss_pdf_cython, data)

# In[31]:
minuit = iminuit.Minuit(unbinned_likelihood, sigma=2, pedantic=False, print_level=0)
# Remember: minuit.errordef is automatically set to 0.5
# as required for likelihood fits (this was explained above)
minuit.migrad() # yes: amazingly fast
unbinned_likelihood.show(minuit)
minuit.print_fmin()
minuit.print_matrix() 

# Out[31]:
# image file: tutorial_files/tutorial_fig_10.png

# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 27927.1139471</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 69</td>
#                 <td title="Number of call in last migrad">NCALLS = 69</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 5.05909350517e-09</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 5e-06</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 0.5</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#OAPuxIqobo').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>mu</td>
#                 <td>9.262679e-01</td>
#                 <td>3.950226e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>sigma</td>
#                 <td>3.950224e+00</td>
#                 <td>2.793227e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="OAPuxIqobo" style="display:none;">
#             <textarea rows="10" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & $\mu$ & 9.263e-01 & 3.950e-02 &  &  &  &  & \\
# \hline
# 2 & $\sigma$ & 3.950e+00 & 2.793e-02 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# 
#             <table>
#                 <tr>
#                     <td><a onclick="$('#PjSOetiEHT').toggle()" href="#">+</a></td>
#         
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             mu
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             sigma
#             </div>
#             </div>
#             </td>
#             
#                 </tr>
#                 
#             <tr>
#                 <td>mu</td>
#             
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>sigma</td>
#             
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#             </tr>
#             </table>
# 
#             <pre id="PjSOetiEHT" style="display:none;">
#             <textarea rows="13" cols="50" onclick="this.select()" readonly>%\usepackage[table]{xcolor} % include this for color
# %\usepackage{rotating} % include this for rotate header
# %\documentclass[xcolor=table]{beamer} % for beamer
# \begin{tabular}{|c|c|c|}
# \hline
# \rotatebox{90}{} & \rotatebox{90}{$\mu$} & \rotatebox{90}{$\sigma$}\\
# \hline
# $\mu$ & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{163,254,186} 0.00\\
# \hline
# $\sigma$ & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{255,117,117} 1.00\\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# In[32]:
# Remember how slow draw_mnprofile() was in the last example?
# Now it's super fast (even though the unbinned
# likelihood computation is more compute-intensive).
minuit.draw_mnprofile('mu');

# Out[32]:
# image file: tutorial_files/tutorial_fig_11.png

# But you really don't have to write your own gaussian, there are tons of builtin functions written in Cython for you.

# In[33]:
# Here's how you can list them
import probfit.pdf
print(dir(probfit.pdf))
print(iminuit.describe(probfit.pdf.gaussian))
print(type(probfit.pdf.gaussian))
# But actually they are always all imported into the main probfit
# namespace, so we'll keep using the simpler probfit.gaussian instead of
# probfit.pdf.gaussian here.

# Out[33]:
#     ['HistogramPdf', 'MinimalFuncCode', 'Polynomial', '_Linear', '__builtins__', '__doc__', '__file__', '__name__', '__package__', '__pyx_capi__', '__test__', 'argus', 'cauchy', 'cruijff', 'crystalball', 'describe', 'doublegaussian', 'gaussian', 'linear', 'novosibirsk', 'np', 'poly2', 'poly3', 'rtv_breitwigner', 'ugaussian']
#     ['x', 'mean', 'sigma']
#     <type 'builtin_function_or_method'>
# 
# In[34]:
unbinned_likelihood = probfit.UnbinnedLH(probfit.gaussian, data)
minuit = iminuit.Minuit(unbinned_likelihood, sigma=2, pedantic=False)
# Remember: minuit.errordef is automatically set to 0.5
# as required for likelihood fits (this was explained above)
minuit.migrad() # yes: amazingly fast
unbinned_likelihood.draw(minuit, show_errbars='normal') # control how fit is displayed too;

# Out[34]:
# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 27927.1139471</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 69</td>
#                 <td title="Number of call in last migrad">NCALLS = 69</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 5.07834778662e-09</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 5e-06</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 0.5</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#bzpnamJMHN').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>mean</td>
#                 <td>9.262679e-01</td>
#                 <td>3.950226e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>sigma</td>
#                 <td>3.950224e+00</td>
#                 <td>2.793227e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="bzpnamJMHN" style="display:none;">
#             <textarea rows="10" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & mean & 9.263e-01 & 3.950e-02 &  &  &  &  & \\
# \hline
# 2 & $\sigma$ & 3.950e+00 & 2.793e-02 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# image file: tutorial_files/tutorial_fig_12.png

# In[35]:
# Draw the difference between data and PDF
plt.figure(figsize=(13,4))
plt.subplot(121)
unbinned_likelihood.draw_residual(minuit)
plt.subplot(122)
unbinned_likelihood.draw_residual(minuit, show_errbars=True, errbar_algo='sumw2', norm=True)

# Out[35]:
# image file: tutorial_files/tutorial_fig_13.png

# ##But... We can't normalize everything analytically and how to generate toy sample from PDF
# 
# When fitting distribution to a PDF, one of the common problem that we run into is normalization.
# Not all function is analytically integrable on the range of our interest.
# 
# Let's look at an example: the [Crystal Ball function](http://en.wikipedia.org/wiki/Crystal_Ball_function).
# It's simply a gaussian with a power law tail ... normally found in energy deposited in crystals ...
# impossible to normalize analytically and normalization will depend on shape parameters.

# In[36]:
numpy.random.seed(0)
bound = (-1, 2)
data = probfit.gen_toy(probfit.crystalball, 10000, bound=bound, alpha=1., n=2., mean=1., sigma=0.3, quiet=False)
# quiet=False tells gen_toy to plot out original function
# toy histogram and poisson error from both orignal distribution and toy

# Out[36]:
#     ['x', 'alpha', 'n', 'mean', 'sigma']
# 
# image file: tutorial_files/tutorial_fig_14.png

# In[37]:
# To fit this function as a distribution we need to normalize
# so that is becomes a PDF ober the range we consider here.
# We do this with the probfit.Normalized functor, which implements
# the trapezoid numerical integration method with a simple cache mechanism
normalized_crystalball = probfit.Normalized(probfit.crystalball, bound)
# this can also bedone with decorator
# @probfit.normalized(bound)
# def my_function(x, blah):
#    return something
pars = 1.0, 1, 2, 1, 0.3
print('function: {}'.format(probfit.crystalball(*pars)))
print('     pdf: {}'.format(normalized_crystalball(*pars)))

# Out[37]:
#     function: 1.0
#          pdf: 1.10945669814
# 
# In[38]:
# The normalized version has the same signature as the non-normalized version
print(iminuit.describe(probfit.crystalball))
print(iminuit.describe(normalized_crystalball))

# Out[38]:
#     ['x', 'alpha', 'n', 'mean', 'sigma']
#     ['x', 'alpha', 'n', 'mean', 'sigma']
# 
# In[39]:
# We can fit the normalized function in the usual way ...
unbinned_likelihood = probfit.UnbinnedLH(normalized_crystalball, data)
start_pars = dict(alpha=1, n=2.1, mean=1.2, sigma=0.3)
minuit = iminuit.Minuit(unbinned_likelihood, **start_pars)
# Remember: minuit.errordef is automatically set to 0.5
# as required for likelihood fits (this was explained above)
minuit.migrad() # yes: amazingly fast Normalize is written in Cython
unbinned_likelihood.show(minuit)
# The Crystal Ball function is notorious for its sensitivity on the 'n' parameter
# probfit give you a heads up where it might have float overflow;

# Out[39]:
#     -c:4: InitialParamWarning: Parameter alpha is floating but does not have initial step size. Assume 1.
#     -c:4: InitialParamWarning: Parameter n is floating but does not have initial step size. Assume 1.
#     -c:4: InitialParamWarning: Parameter mean is floating but does not have initial step size. Assume 1.
#     -c:4: InitialParamWarning: Parameter sigma is floating but does not have initial step size. Assume 1.
#     -c:7: SmallIntegralWarning: (0.9689428295957161, 0.44086027281289175, -7.852819454184058, 0.9263111214440007, 0.29811644525305303)
# 
# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 6154.37579109</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 178</td>
#                 <td title="Number of call in last migrad">NCALLS = 178</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 1.09346528365e-06</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 5e-06</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 0.5</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#YWsxkHCEHl').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>alpha</td>
#                 <td>1.012962e+00</td>
#                 <td>5.321735e-02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>n</td>
#                 <td>1.812763e+00</td>
#                 <td>2.177144e-01</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>3</td>
#                 <td>mean</td>
#                 <td>9.982474e-01</td>
#                 <td>5.583931e-03</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>4</td>
#                 <td>sigma</td>
#                 <td>2.996611e-01</td>
#                 <td>4.195338e-03</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="YWsxkHCEHl" style="display:none;">
#             <textarea rows="14" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & $\alpha$ & 1.013e+00 & 5.322e-02 &  &  &  &  & \\
# \hline
# 2 & n & 1.813e+00 & 2.177e-01 &  &  &  &  & \\
# \hline
# 3 & mean & 9.982e-01 & 5.584e-03 &  &  &  &  & \\
# \hline
# 4 & $\sigma$ & 2.997e-01 & 4.195e-03 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# image file: tutorial_files/tutorial_fig_15.png

# ## But what if I know the analytical integral formula for my distribution?
# 
# `probfit` checks for a method called `integrate` with the signature `integrate(bound, nint, *arg)` to
# compute definite integrals for given `bound` and `nint` (pieces of integral this is normally ignored)
# and the rest will be passed as positional argument.
# 
# For some `probfit` built-in distributions analytical formulae have been implemented.

# In[40]:
def line(x, m, c):
    return m * x + c

# compute integral of line from x=(0,1) using 10 intevals with m=1. and c=2.
# all probfit internal use this
# no integrate method available probfit use simpson3/8
print(probfit.integrate1d(line, (0, 1), 10, (1., 2.)))

# Let us illustrate the point by forcing it to have integral that's off by
# factor of two
def wrong_line_integrate(bound, nint, m, c):
    a, b = bound
    # I know this is wrong:
    return 2 * (m * (b ** 2 / 2. - a ** 2 / 2.) + c * (b - a))

line.integrate = wrong_line_integrate
# line.integrate = lambda bound, nint, m, c: blah blah # this works too
print(probfit.integrate1d(line, (0, 1), 10, (1., 2.)))

# Out[40]:
#     2.5
#     5.0
# 
### What if things go wrong?

# In this section we show you what happens when your distribution doesn't fit and how you can make it.
# 
# We again use the Crystal Ball distribution as an example, which is notoriously sensitive to initial parameter values.

# In[41]:
unbinned_likelihood = probfit.UnbinnedLH(normalized_crystalball, data)
# No initial values given -> all parameters have default initial value 0
minuit = iminuit.Minuit(unbinned_likelihood)
# Remember: minuit.errordef is automatically set to 0.5
# as required for likelihood fits (this was explained above)
minuit.migrad() # yes: amazingly fast but tons of output on the console
# Remember there is a heads up;

# Out[41]:
#     -c:3: InitialParamWarning: Parameter alpha does not have initial value. Assume 0.
#     -c:3: InitialParamWarning: Parameter alpha is floating but does not have initial step size. Assume 1.
#     -c:3: InitialParamWarning: Parameter n does not have initial value. Assume 0.
#     -c:3: InitialParamWarning: Parameter n is floating but does not have initial step size. Assume 1.
#     -c:3: InitialParamWarning: Parameter mean does not have initial value. Assume 0.
#     -c:3: InitialParamWarning: Parameter mean is floating but does not have initial step size. Assume 1.
#     -c:3: InitialParamWarning: Parameter sigma does not have initial value. Assume 0.
#     -c:3: InitialParamWarning: Parameter sigma is floating but does not have initial step size. Assume 1.
# 
# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 10986.1228867</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 230</td>
#                 <td title="Number of call in last migrad">NCALLS = 230</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 0.0</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 5e-06</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 0.5</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#FF7878">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#FF7878">False</td>
#                 <td align="center" style="background-color:#FF7878">False</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#FF7878">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#lxHAcKWNeN').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>alpha</td>
#                 <td>0.000000e+00</td>
#                 <td>1.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>n</td>
#                 <td>0.000000e+00</td>
#                 <td>1.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>3</td>
#                 <td>mean</td>
#                 <td>0.000000e+00</td>
#                 <td>1.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>4</td>
#                 <td>sigma</td>
#                 <td>0.000000e+00</td>
#                 <td>1.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="lxHAcKWNeN" style="display:none;">
#             <textarea rows="14" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & $\alpha$ & 0.000e+00 & 1.000e+00 &  &  &  &  & \\
# \hline
# 2 & n & 0.000e+00 & 1.000e+00 &  &  &  &  & \\
# \hline
# 3 & mean & 0.000e+00 & 1.000e+00 &  &  &  &  & \\
# \hline
# 4 & $\sigma$ & 0.000e+00 & 1.000e+00 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# In[42]:
# This shows that we failed.
# The parameters are still at the default initial values
unbinned_likelihood.show(minuit);

# Out[42]:
# image file: tutorial_files/tutorial_fig_16.png

# In[43]:
# These two status flags tell you if the best-fit parameter values
# and the covariance matrix (the parameter errors) are OK.
print(minuit.migrad_ok())
print(minuit.matrix_accurate())

# Out[43]:
#     False
#     False
# 
# To make MIGRAD converge we need start parameter values that are roughly correct. Remember that above the same fit converged when we used ::
# 
#     start_pars = dict(alpha=1, n=2.1, mean=1.2, sigma=0.3)
#     minuit = iminuit.Minuit(unbinned_likelihood, **start_pars)
#     
# #### But how can we guess these initial values?
# 
# This is a hard question that doesn't have one simple answer. Visualizing your data and model helps.
# 

# In[44]:
# Try one set of parameters
best_try = probfit.try_uml(normalized_crystalball, data, alpha=1., n=2.1, mean=1.2, sigma=0.3)
print(best_try)

# Out[44]:
#     {'alpha': 1.0, 'mean': 1.2, 'sigma': 0.3, 'n': 2.1}
# 
# image file: tutorial_files/tutorial_fig_17.png

# In[45]:
# Or try multiple sets of parameters
# (too many will just confuse you)
best_try = probfit.try_uml(normalized_crystalball, data, alpha=1., n=2.1, mean=[1.2, 1.1], sigma=[0.3, 0.5])
# try_uml computes the unbinned likelihood for each set of parameters and returns the best
# one as a dictionary.
# This is actually a poor-man's optimization algorithm in itself called grid search
# which is popular to find good start values for other, faster optimization methods like MIGRAD.
print(best_try)

# Out[45]:
#     {'alpha': 1.0, 'mean': 1.1, 'sigma': 0.3, 'n': 2.1}
# 
# image file: tutorial_files/tutorial_fig_18.png

### Extended fit: two Gaussians with polynomial background

# Here we show how to create and fit a model that is the sum of several other models.

# In[46]:
# Generate some example data
np.random.seed(0)
data_peak1 = np.random.randn(3000) * 0.2 + 2
data_peak2 = np.random.randn(5000) * 0.1 + 4
data_range = (-2, 5)
data_bg = probfit.gen_toy(lambda x : 4 + 4 * x + x ** 2, 20000, data_range)
data_all = np.concatenate([data_peak1, data_peak2, data_bg])
plt.hist((data_peak1, data_peak2, data_bg, data_all),
         label=['Signal 1', 'Signal 2', 'Background', 'Total'],
         bins=200, histtype='step', range=data_range)
plt.legend(loc='upper left');

# Out[46]:
# image file: tutorial_files/tutorial_fig_19.png

# In[47]:
# Using a polynomial to fit a distribution is problematic, because the
# polynomial can assume negative values, which results in NaN (not a number)
# values in the likelihood function.
# To avoid this problem we restrict the fit to the range (0, 5) where
# the polynomial is clearly positive.
fit_range = (0, 5)
normalized_poly = probfit.Normalized(probfit.Polynomial(2), fit_range)
normalized_poly = probfit.Extended(normalized_poly, extname='NBkg')

gauss1 = probfit.Extended(probfit.rename(probfit.gaussian, ['x', 'mu1', 'sigma1']), extname='N1')
gauss2 = probfit.Extended(probfit.rename(probfit.gaussian, ['x', 'mu2', 'sigma2']), extname='N2')

# Define an extended PDF consisting of three components
pdf = probfit.AddPdf(normalized_poly, gauss1, gauss2)

print('normalized_poly: {}'.format(probfit.describe(normalized_poly)))
print('gauss1:          {}'.format(probfit.describe(gauss1)))
print('gauss2:          {}'.format(probfit.describe(gauss2)))
print('pdf:             {}'.format(probfit.describe(pdf)))

# Out[47]:
#     normalized_poly: ['x', 'c_0', 'c_1', 'c_2', 'NBkg']
#     gauss1:          ['x', 'mu1', 'sigma1', 'N1']
#     gauss2:          ['x', 'mu2', 'sigma2', 'N2']
#     pdf:             ['x', 'c_0', 'c_1', 'c_2', 'NBkg', 'mu1', 'sigma1', 'N1', 'mu2', 'sigma2', 'N2']
# 
# In[48]:
# Define the cost function in the usual way ...
binned_likelihood = probfit.BinnedLH(pdf, data_all, bins=200, extended=True, bound=fit_range)

# This is a quite complex fit (11 free parameters!), so we need good starting values.
# Actually we even need to set an initial parameter error
# for 'mu1' and 'mu2' to make MIGRAD converge.
# The initial parameter error is used as the initial step size in the minimization.
pars = dict(mu1=1.9, error_mu1=0.1, sigma1=0.2, N1=3000,
            mu2=4.1, error_mu2=0.1, sigma2=0.1, N2=5000,
            c_0=4, c_1=4, c_2=1, NBkg=20000)
minuit = iminuit.Minuit(binned_likelihood, pedantic=False, print_level=0, **pars)
# You can see that the model already roughly matches the data
binned_likelihood.draw(minuit, parts=True);

# Out[48]:
# image file: tutorial_files/tutorial_fig_20.png

# In[49]:
# This can take a while ... the likelihood is evaluated a few 100 times
# (and each time the distributions are evaluated, including the
# numerical computation of the normalizing integrals)
minuit.migrad();

# In[50]:
binned_likelihood.show(minuit, parts=True);
minuit.print_fmin()
minuit.print_matrix()

# Out[50]:
# image file: tutorial_files/tutorial_fig_21.png

# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 88.5482069321</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 331</td>
#                 <td title="Number of call in last migrad">NCALLS = 331</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 2.50658801602e-06</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 5e-06</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 0.5</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#pQMVDPBOiT').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>c_0</td>
#                 <td>4.156499e+00</td>
#                 <td>3.275183e+00</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>c_1</td>
#                 <td>3.761296e+00</td>
#                 <td>3.028133e+00</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>3</td>
#                 <td>c_2</td>
#                 <td>9.724907e-01</td>
#                 <td>7.669779e-01</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>4</td>
#                 <td>NBkg</td>
#                 <td>1.960121e+04</td>
#                 <td>1.745707e+02</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>5</td>
#                 <td>mu1</td>
#                 <td>1.990820e+00</td>
#                 <td>5.813670e-03</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>6</td>
#                 <td>sigma1</td>
#                 <td>1.921099e-01</td>
#                 <td>5.825608e-03</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>7</td>
#                 <td>N1</td>
#                 <td>2.923541e+03</td>
#                 <td>9.135432e+01</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>8</td>
#                 <td>mu2</td>
#                 <td>3.994147e+00</td>
#                 <td>2.123760e-03</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>9</td>
#                 <td>sigma2</td>
#                 <td>1.001096e-01</td>
#                 <td>1.999429e-03</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>10</td>
#                 <td>N2</td>
#                 <td>4.969156e+03</td>
#                 <td>9.797528e+01</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="pQMVDPBOiT" style="display:none;">
#             <textarea rows="26" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & $c_{0}$ & 4.156e+00 & 3.275e+00 &  &  &  &  & \\
# \hline
# 2 & $c_{1}$ & 3.761e+00 & 3.028e+00 &  &  &  &  & \\
# \hline
# 3 & $c_{2}$ & 9.725e-01 & 7.670e-01 &  &  &  &  & \\
# \hline
# 4 & NBkg & 1.960e+04 & 1.746e+02 &  &  &  &  & \\
# \hline
# 5 & mu1 & 1.991e+00 & 5.814e-03 &  &  &  &  & \\
# \hline
# 6 & sigma1 & 1.921e-01 & 5.826e-03 &  &  &  &  & \\
# \hline
# 7 & N1 & 2.924e+03 & 9.135e+01 &  &  &  &  & \\
# \hline
# 8 & mu2 & 3.994e+00 & 2.124e-03 &  &  &  &  & \\
# \hline
# 9 & sigma2 & 1.001e-01 & 1.999e-03 &  &  &  &  & \\
# \hline
# 10 & N2 & 4.969e+03 & 9.798e+01 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# 
#             <table>
#                 <tr>
#                     <td><a onclick="$('#OKHuWoxfdj').toggle()" href="#">+</a></td>
#         
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             c_0
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             c_1
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             c_2
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             NBkg
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             mu1
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             sigma1
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             N1
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             mu2
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             sigma2
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             N2
#             </div>
#             </div>
#             </td>
#             
#                 </tr>
#                 
#             <tr>
#                 <td>c_0</td>
#             
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(253,120,118)">
#                 0.98
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,185)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,185)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 0.00
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>c_1</td>
#             
#                 <td style="background-color:rgb(253,120,118)">
#                 0.98
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(252,121,119)">
#                 0.97
#                 </td>
#                 
#                 <td style="background-color:rgb(167,248,183)">
#                 0.04
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,186)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(169,245,182)">
#                 -0.06
#                 </td>
#                 
#                 <td style="background-color:rgb(171,242,180)">
#                 -0.09
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,185)">
#                 0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 0.01
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>c_2</td>
#             
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(252,121,119)">
#                 0.97
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,185)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 -0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(165,252,185)">
#                 0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(165,251,184)">
#                 0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 -0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,186)">
#                 -0.01
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>NBkg</td>
#             
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(167,248,183)">
#                 0.04
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,185)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(167,248,183)">
#                 -0.05
#                 </td>
#                 
#                 <td style="background-color:rgb(189,215,166)">
#                 -0.28
#                 </td>
#                 
#                 <td style="background-color:rgb(197,204,161)">
#                 -0.37
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(185,222,170)">
#                 -0.23
#                 </td>
#                 
#                 <td style="background-color:rgb(190,213,166)">
#                 -0.30
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>mu1</td>
#             
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,186)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 -0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(167,248,183)">
#                 -0.05
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(170,243,181)">
#                 0.08
#                 </td>
#                 
#                 <td style="background-color:rgb(169,245,181)">
#                 0.07
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(164,252,185)">
#                 0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(165,251,185)">
#                 0.02
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>sigma1</td>
#             
#                 <td style="background-color:rgb(164,253,185)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(169,245,182)">
#                 -0.06
#                 </td>
#                 
#                 <td style="background-color:rgb(165,252,185)">
#                 0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(189,215,166)">
#                 -0.28
#                 </td>
#                 
#                 <td style="background-color:rgb(170,243,181)">
#                 0.08
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(209,185,151)">
#                 0.50
#                 </td>
#                 
#                 <td style="background-color:rgb(164,252,185)">
#                 -0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(166,250,184)">
#                 0.03
#                 </td>
#                 
#                 <td style="background-color:rgb(166,249,183)">
#                 0.04
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>N1</td>
#             
#                 <td style="background-color:rgb(164,253,185)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(171,242,180)">
#                 -0.09
#                 </td>
#                 
#                 <td style="background-color:rgb(165,251,184)">
#                 0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(197,204,161)">
#                 -0.37
#                 </td>
#                 
#                 <td style="background-color:rgb(169,245,181)">
#                 0.07
#                 </td>
#                 
#                 <td style="background-color:rgb(209,185,151)">
#                 0.50
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(165,251,185)">
#                 -0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(166,249,183)">
#                 0.04
#                 </td>
#                 
#                 <td style="background-color:rgb(167,247,183)">
#                 0.05
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>mu2</td>
#             
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,185)">
#                 0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 -0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(164,252,185)">
#                 -0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(165,251,185)">
#                 -0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(166,249,184)">
#                 0.03
#                 </td>
#                 
#                 <td style="background-color:rgb(166,250,184)">
#                 0.03
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>sigma2</td>
#             
#                 <td style="background-color:rgb(163,253,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(185,222,170)">
#                 -0.23
#                 </td>
#                 
#                 <td style="background-color:rgb(164,252,185)">
#                 0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(166,250,184)">
#                 0.03
#                 </td>
#                 
#                 <td style="background-color:rgb(166,249,183)">
#                 0.04
#                 </td>
#                 
#                 <td style="background-color:rgb(166,249,184)">
#                 0.03
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(198,202,160)">
#                 0.38
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>N2</td>
#             
#                 <td style="background-color:rgb(163,253,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,253,186)">
#                 0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(164,253,186)">
#                 -0.01
#                 </td>
#                 
#                 <td style="background-color:rgb(190,213,166)">
#                 -0.30
#                 </td>
#                 
#                 <td style="background-color:rgb(165,251,185)">
#                 0.02
#                 </td>
#                 
#                 <td style="background-color:rgb(166,249,183)">
#                 0.04
#                 </td>
#                 
#                 <td style="background-color:rgb(167,247,183)">
#                 0.05
#                 </td>
#                 
#                 <td style="background-color:rgb(166,250,184)">
#                 0.03
#                 </td>
#                 
#                 <td style="background-color:rgb(198,202,160)">
#                 0.38
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#             </tr>
#             </table>
# 
#             <pre id="OKHuWoxfdj" style="display:none;">
#             <textarea rows="29" cols="50" onclick="this.select()" readonly>%\usepackage[table]{xcolor} % include this for color
# %\usepackage{rotating} % include this for rotate header
# %\documentclass[xcolor=table]{beamer} % for beamer
# \begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}
# \hline
# \rotatebox{90}{} & \rotatebox{90}{$c_{0}$} & \rotatebox{90}{$c_{1}$} & \rotatebox{90}{$c_{2}$} & \rotatebox{90}{NBkg} & \rotatebox{90}{mu1} & \rotatebox{90}{sigma1} & \rotatebox{90}{N1} & \rotatebox{90}{mu2} & \rotatebox{90}{sigma2} & \rotatebox{90}{N2}\\
# \hline
# $c_{0}$ & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{253,120,118} 0.98 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{164,253,185} -0.01 & \cellcolor[RGB]{164,253,185} -0.01 & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{163,253,186} 0.00 & \cellcolor[RGB]{163,253,186} 0.00\\
# \hline
# $c_{1}$ & \cellcolor[RGB]{253,120,118} 0.98 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{252,121,119} 0.97 & \cellcolor[RGB]{167,248,183} 0.04 & \cellcolor[RGB]{164,253,186} -0.01 & \cellcolor[RGB]{169,245,182} -0.06 & \cellcolor[RGB]{171,242,180} -0.09 & \cellcolor[RGB]{164,253,185} 0.01 & \cellcolor[RGB]{163,253,186} 0.00 & \cellcolor[RGB]{163,253,186} 0.01\\
# \hline
# $c_{2}$ & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{252,121,119} 0.97 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{164,253,185} -0.01 & \cellcolor[RGB]{163,254,186} -0.00 & \cellcolor[RGB]{165,252,185} 0.02 & \cellcolor[RGB]{165,251,184} 0.02 & \cellcolor[RGB]{163,254,186} -0.00 & \cellcolor[RGB]{163,253,186} -0.01 & \cellcolor[RGB]{164,253,186} -0.01\\
# \hline
# NBkg & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{167,248,183} 0.04 & \cellcolor[RGB]{164,253,185} -0.01 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{167,248,183} -0.05 & \cellcolor[RGB]{189,215,166} -0.28 & \cellcolor[RGB]{197,204,161} -0.37 & \cellcolor[RGB]{163,253,186} -0.01 & \cellcolor[RGB]{185,222,170} -0.23 & \cellcolor[RGB]{190,213,166} -0.30\\
# \hline
# mu1 & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{164,253,186} -0.01 & \cellcolor[RGB]{163,254,186} -0.00 & \cellcolor[RGB]{167,248,183} -0.05 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{170,243,181} 0.08 & \cellcolor[RGB]{169,245,181} 0.07 & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{164,252,185} 0.02 & \cellcolor[RGB]{165,251,185} 0.02\\
# \hline
# sigma1 & \cellcolor[RGB]{164,253,185} -0.01 & \cellcolor[RGB]{169,245,182} -0.06 & \cellcolor[RGB]{165,252,185} 0.02 & \cellcolor[RGB]{189,215,166} -0.28 & \cellcolor[RGB]{170,243,181} 0.08 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{209,185,151} 0.50 & \cellcolor[RGB]{164,252,185} -0.02 & \cellcolor[RGB]{166,250,184} 0.03 & \cellcolor[RGB]{166,249,183} 0.04\\
# \hline
# N1 & \cellcolor[RGB]{164,253,185} -0.01 & \cellcolor[RGB]{171,242,180} -0.09 & \cellcolor[RGB]{165,251,184} 0.02 & \cellcolor[RGB]{197,204,161} -0.37 & \cellcolor[RGB]{169,245,181} 0.07 & \cellcolor[RGB]{209,185,151} 0.50 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{165,251,185} -0.02 & \cellcolor[RGB]{166,249,183} 0.04 & \cellcolor[RGB]{167,247,183} 0.05\\
# \hline
# mu2 & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{164,253,185} 0.01 & \cellcolor[RGB]{163,254,186} -0.00 & \cellcolor[RGB]{163,253,186} -0.01 & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{164,252,185} -0.02 & \cellcolor[RGB]{165,251,185} -0.02 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{166,249,184} 0.03 & \cellcolor[RGB]{166,250,184} 0.03\\
# \hline
# sigma2 & \cellcolor[RGB]{163,253,186} 0.00 & \cellcolor[RGB]{163,253,186} 0.00 & \cellcolor[RGB]{163,253,186} -0.01 & \cellcolor[RGB]{185,222,170} -0.23 & \cellcolor[RGB]{164,252,185} 0.02 & \cellcolor[RGB]{166,250,184} 0.03 & \cellcolor[RGB]{166,249,183} 0.04 & \cellcolor[RGB]{166,249,184} 0.03 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{198,202,160} 0.38\\
# \hline
# N2 & \cellcolor[RGB]{163,253,186} 0.00 & \cellcolor[RGB]{163,253,186} 0.01 & \cellcolor[RGB]{164,253,186} -0.01 & \cellcolor[RGB]{190,213,166} -0.30 & \cellcolor[RGB]{165,251,185} 0.02 & \cellcolor[RGB]{166,249,183} 0.04 & \cellcolor[RGB]{167,247,183} 0.05 & \cellcolor[RGB]{166,250,184} 0.03 & \cellcolor[RGB]{198,202,160} 0.38 & \cellcolor[RGB]{255,117,117} 1.00\\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# Note the red upper left corner in the correlation matrix above?
# 
# It shows that the three polynomial parameters `c_0`, `c_1` and `c_2` are highly correlated?
# The reason is that we put a constraint on the polynomial to be normalized over the fit range:
# 
#     fit_range = (0, 5)
#     normalized_poly = probfit.Normalized(probfit.Polynomial(2), fit_range)
#     normalized_poly = probfit.Extended(normalized_poly, extname='NBkg')
# 
# To resolve this problem you could simply use a non-normalized and non-extended polynomial to model the background. We won't do this here, though ...

# ## Custom Drawing
# 
# The `draw()` and `show()` method we provide is intended to just give you a quick look at your fit.
# 
# To make a custom drawing you can use the return value of `draw()` and `show()`.

# In[51]:
# You should copy & paste the return tuple from the `draw` docstring ...
((data_edges, datay), (errorp, errorm), (total_pdf_x, total_pdf_y), parts) = binned_likelihood.draw(minuit, parts=True);
# ... now we have everything to make our own plot

# Out[51]:
# image file: tutorial_files/tutorial_fig_22.png

# In[52]:
# Now make the plot as pretty as you like, e.g. with matplotlib.
plt.figure(figsize=(8, 5))
plt.errorbar(probfit.mid(data_edges), datay, errorp, fmt='.', capsize=0, color='Gray', label='Data')
plt.plot(total_pdf_x, total_pdf_y, color='blue', lw=2, label='Total Model')
colors = ['orange', 'purple', 'DarkGreen']
labels = ['Background', 'Signal 1', 'Signal 2']
for color, label, part in zip(colors, labels, parts):
    x, y = part
    plt.plot(x, y, ls='--', color=color, label=label)
plt.grid(True)
plt.legend(loc='upper left');

# Out[52]:
# image file: tutorial_files/tutorial_fig_23.png

# ## Simultaneous fit to several data sets
# 
# Sometimes, what we want to fit is the sum of likelihood /chi^2 of two PDFs for two different datasets that share some parameters.
# 
# In this example, we will fit two Gaussian distributions where we know that the widths are the same
# but the peaks are at different places.

# In[53]:
# Generate some example data
np.random.seed(0)
data1 = np.random.randn(10000) + 3 # mean =  3, sigma = 1
data2 = np.random.randn(10000) - 2 # mean = -2, sigma = 1
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.hist(data1, bins=100, range=(-7, 7), histtype='step', label='data1')
plt.legend()
plt.subplot(122)
plt.hist(data2, bins=100, range=(-7, 7), histtype='step', label='data2')
plt.legend();

# Out[53]:
# image file: tutorial_files/tutorial_fig_24.png

# In[54]:
# There is nothing special about built-in cost function
# except some utility function like draw and show
likelihood1 = probfit.UnbinnedLH(probfit.rename(probfit.gaussian, ('x', 'mean2', 'sigma')), data1)
likelihood2 = probfit.UnbinnedLH(probfit.gaussian, data2)
simultaneous_likelihood = probfit.SimultaneousFit(likelihood1, likelihood2)
print(probfit.describe(likelihood1))
print(probfit.describe(likelihood2))
# Note that the simultaneous likelihood has only 3 parameters, because the
# 'sigma' parameter is tied (i.e. linked to always be the same).
print(probfit.describe(simultaneous_likelihood))

# Out[54]:
#     ['mean2', 'sigma']
#     ['mean', 'sigma']
#     ['mean2', 'sigma', 'mean']
# 
# In[55]:
# Ah, the beauty of Minuit ... it doesn't care what your cost funtion is ...
# you can use it to fit (i.e. compute optimal parameters and parameter errors) anything.
minuit = iminuit.Minuit(simultaneous_likelihood, sigma=0.5, pedantic=False, print_level=0)
# Well, there's one thing we have to tell Minuit so that it can compute parameter errors,
# and that is the value of `errordef`, a.k.a. `up` (explained above).
# This is a likelihood fit, so we need `errordef = 0.5` and not the default `errordef = 1`:
minuit.errordef = 0.5

# In[56]:
# Run the fit and print the results
minuit.migrad();
minuit.print_fmin()
minuit.print_matrix()

# Out[56]:
# <hr>
# 
#         <table>
#             <tr>
#                 <td title="Minimum value of function">FCN = 28184.0142876</td>
#                 <td title="Total number of call to FCN so far">TOTAL NCALL = 97</td>
#                 <td title="Number of call in last migrad">NCALLS = 97</td>
#             </tr>
#             <tr>
#                 <td title="Estimated distance to minimum">EDM = 2.24660525589e-09</td>
#                 <td title="Maximum EDM definition of convergence">GOAL EDM = 5e-06</td>
#                 <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
#                 UP = 0.5</td>
#             </tr>
#         </table>
#         
#         <table>
#             <tr>
#                 <td align="center" title="Validity of the migrad call">Valid</td>
#                 <td align="center" title="Validity of parameters">Valid Param</td>
#                 <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
#                 <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
#                 <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#             <tr>
#                 <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
#                 <td align="center" title="Validity of covariance">HasCov</td>
#                 <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
#                 <td align="center"></td>
#                 <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
#             </tr>
#             <tr>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center" style="background-color:#92CCA6">True</td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#                 <td align="center"></td>
#                 <td align="center" style="background-color:#92CCA6">False</td>
#             </tr>
#         </table>
#         
# 
#         <table>
#             <tr>
#                 <td><a href="#" onclick="$('#iGsKVySUxS').toggle()">+</a></td>
#                 <td title="Variable name">Name</td>
#                 <td title="Value of parameter">Value</td>
#                 <td title="Parabolic error">Parab Error</td>
#                 <td title="Minos lower error">Minos Error-</td>
#                 <td title="Minos upper error">Minos Error+</td>
#                 <td title="Lower limit of the parameter">Limit-</td>
#                 <td title="Upper limit of the parameter">Limit+</td>
#                 <td title="Is the parameter fixed in the fit">FIXED</td>
#             </tr>
#         
#             <tr>
#                 <td>1</td>
#                 <td>mean2</td>
#                 <td>2.981566e+00</td>
#                 <td>9.903099e-03</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>2</td>
#                 <td>sigma</td>
#                 <td>9.903098e-01</td>
#                 <td>4.951551e-03</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             <tr>
#                 <td>3</td>
#                 <td>mean</td>
#                 <td>-1.989012e+00</td>
#                 <td>9.903099e-03</td>
#                 <td>0.000000e+00</td>
#                 <td>0.000000e+00</td>
#                 <td></td>
#                 <td></td>
#                 <td></td>
#             </tr>
#             
#             </table>
#         
#             <pre id="iGsKVySUxS" style="display:none;">
#             <textarea rows="12" cols="50" onclick="this.select()" readonly>\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
# \hline
#  & Name & Value & Para Error & Error+ & Error- & Limit+ & Limit- & FIXED\\
# \hline
# 1 & mean2 & 2.982e+00 & 9.903e-03 &  &  &  &  & \\
# \hline
# 2 & $\sigma$ & 9.903e-01 & 4.952e-03 &  &  &  &  & \\
# \hline
# 3 & mean & -1.989e+00 & 9.903e-03 &  &  &  &  & \\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# <hr>
# 
#             <table>
#                 <tr>
#                     <td><a onclick="$('#ksThmHphBC').toggle()" href="#">+</a></td>
#         
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             mean2
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             sigma
#             </div>
#             </div>
#             </td>
#             
#             <td>
#             <div style="width:20px;position:relative; width: -moz-fit-content;">
#             <div style="display:inline-block;-webkit-writing-mode:vertical-rl;-moz-writing-mode: vertical-rl;writing-mode: vertical-rl;">
#             mean
#             </div>
#             </div>
#             </td>
#             
#                 </tr>
#                 
#             <tr>
#                 <td>mean2</td>
#             
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>sigma</td>
#             
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#             </tr>
#             
#             <tr>
#                 <td>mean</td>
#             
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(163,254,186)">
#                 0.00
#                 </td>
#                 
#                 <td style="background-color:rgb(255,117,117)">
#                 1.00
#                 </td>
#                 
#             </tr>
#             </table>
# 
#             <pre id="ksThmHphBC" style="display:none;">
#             <textarea rows="15" cols="50" onclick="this.select()" readonly>%\usepackage[table]{xcolor} % include this for color
# %\usepackage{rotating} % include this for rotate header
# %\documentclass[xcolor=table]{beamer} % for beamer
# \begin{tabular}{|c|c|c|c|}
# \hline
# \rotatebox{90}{} & \rotatebox{90}{mean2} & \rotatebox{90}{$\sigma$} & \rotatebox{90}{mean}\\
# \hline
# mean2 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{163,254,186} 0.00\\
# \hline
# $\sigma$ & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{163,254,186} 0.00\\
# \hline
# mean & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{255,117,117} 1.00\\
# \hline
# \end{tabular}</textarea>
#             </pre>
#             
# In[57]:
simultaneous_likelihood.draw(minuit);

# Out[57]:
# image file: tutorial_files/tutorial_fig_25.png

# In[58]:
# Now it's your turn ...
# try and apply probfit / iminuit and to your modeling / fitting task! 
