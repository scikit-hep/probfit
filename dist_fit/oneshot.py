from .cdist_fit import UnbinnedML, BinnedChi2,compute_cdf,invert_cdf,BinnedLH
from RTMinuit import Minuit
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as npr
import itertools as itt
import collections
from util import vertical_highlight
def randfr(r):
    b = r[1]
    a = r[0]
    return np.random.ranf()*(b-a)+a

def fit_uml(f, data, quiet=False, printlevel=0, *arg, **kwd):
    uml = UnbinnedML(f,data)
    m = Minuit(uml,printlevel=printlevel,**kwd)
    m.set_strategy(2)
    m.set_up(0.5)
    m.migrad()
    if not m.migrad_ok() or not m.matrix_accurate():
        if not quiet:
            plt.figure()
            uml.show()
            print m.values
    return (uml,m)

def fit_binx2(f,data,bins=30, range=None, printlevel=0, quiet=False, *arg, **kwd):
    uml = BinnedChi2(f,data,bins=bins,range=range)
    m = Minuit(uml,printlevel=printlevel,**kwd)
    m.set_strategy(2)
    m.set_up(1)
    m.migrad()
    if not m.migrad_ok() or not m.matrix_accurate():
        if not quiet:
            plt.figure()
            uml.show()
            print m.values
    return (uml,m)

def fit_binlh(f,data,bins=30, 
        range=None, quiet=False, weights=None,use_w2=False,
        printlevel=0,pedantic=True,extended=False, 
        *arg, **kwd):
    uml = BinnedLH(f,data,bins=bins,range=range,weights=weights,use_w2=use_w2,extended=extended)
    m = Minuit(uml,printlevel=printlevel,pedantic=pedantic,**kwd)
    m.set_strategy(2)
    m.set_up(0.5)
    m.migrad()
    if not m.migrad_ok() or not m.matrix_accurate():
        if not quiet:
            plt.figure()
            uml.show()
            print m.values
    #m.minos()
    return (uml,m)

def tuplize(x):
    if  isinstance(x, collections.Iterable):
        return x
    else:
        return tuple([x])

def pprint_arg(vnames,value):
    ret = ''
    for name,v in zip(vnames,value):
        ret +='%s=%s;'%(name,str(v))
    return ret;

def try_uml(f,data,bins=40,fbins=1000,*arg,**kwd):
    fom = UnbinnedML(f,data)
    narg = f.func_code.co_argcount
    vnames = f.func_code.co_varnames[1:narg]
    my_arg = [ tuplize(kwd[name]) for name in vnames ]
    h,e,_ = plt.hist(data,bins=bins,normed=True,histtype='step')
    vf = np.vectorize(f)
    vx = np.linspace(e[0],e[-1],fbins)
    first = True
    minfom = 0
    minarg = None
    for thisarg in itt.product(*my_arg):
        vy = vf(vx,*thisarg)
        plt.plot(vx,vy,'-',label=pprint_arg(vnames,thisarg))
        thisfom = fom(*thisarg)
        if first or thisfom<minfom:
            minfom = thisfom
            minarg = thisarg
            first = False
    leg = plt.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ret = {k:v for k,v in zip(vnames,minarg)}
    return ret
    
def try_chi2(f,data,weights=None,bins=40,fbins=1000,show='both',*arg,**kwd):
    fom = BinnedChi2(f,data)
    narg = f.func_code.co_argcount
    vnames = f.func_code.co_varnames[1:narg]
    my_arg = [ tuplize(kwd[name]) for name in vnames ]
    h,e=None,None
    if show=='both':
        h,e,_ = plt.hist(data,bins=bins,histtype='step',weights=weights)
    else:
        h,e = np.histogram(data,bins=bins,weights=weights)
    bw = e[1]-e[0]
    vf = np.vectorize(f)
    vx = np.linspace(e[0],e[-1],fbins)
    first = True
    minfom = 0
    minarg = None
    for thisarg in itt.product(*my_arg):
        vy = vf(vx,*thisarg)*bw
        plt.plot(vx,vy,'-',label=pprint_arg(vnames,thisarg))
        thisfom = fom(*thisarg)
        if first or thisfom<minfom:
            minfom = thisfom
            minarg = thisarg
            first = False
    leg = plt.legend(fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ret = {k:v for k,v in zip(vnames,minarg)}
    return ret

def gen_toy(f,numtoys,range,accuracy=10000,quiet=True,**kwd):
    #based on inverting cdf this is fast but you will need to give it a reasonable range
    #unlike roofit which is based on accept reject
    narg = f.func_code.co_argcount
    vnames = f.func_code.co_varnames[1:narg]
    my_arg = [ kwd[name] for name in vnames ]
    #random number
    #if accuracy is None: accuracy=10*numtoys
    r = npr.random_sample(numtoys)
    x = np.linspace(range[0],range[1],accuracy)
    vf = np.vectorize(f)
    pdf = vf(x,*my_arg)
    if not quiet:
        plt.figure()
        plt.plot(x,pdf)
        plt.title('pdf')
        plt.ylim(ymin=0)
        plt.show()
    cdf = compute_cdf(pdf,x)
    if cdf[-1] < 0.01:
        print 'Integral for given funcition is really low. Did you give it a reasonable range?'
    cdfnorm = cdf[-1]
    cdf/=cdfnorm
    if not quiet:
        plt.figure()
        plt.title('cdf')
        plt.plot(x,cdf)
        plt.show()
    #now convert that to toy
    ret = invert_cdf(r, cdf, x)
    
    if not quiet:
        plt.figure()
        plt.title('comparison')
        numbin = 100
        h,e,_ = plt.hist(ret,bins=numbin,histtype='step',label='generated',color='r')
        bw = e[1]-e[0]
        y = pdf*len(ret)/cdfnorm*bw
        ylow = y+np.sqrt(y)
        yhigh = y-np.sqrt(y)
        plt.plot(x,y,label='pdf',color='b')
        plt.fill_between(x,yhigh,ylow,color='b',alpha=0.2)
    return ret

    
def guess_initial(alg,f,data,ntry=100,guessrange=(-100,100),draw=False,*arg,**kwd):
    """
    This is very bad at the moment don't use it
    """
    fom = alg(f,data,*arg,**kwd)
    first =True
    minfom =0
    minparam=()
    narg = fom.func_code.co_argcount
    vnames = fom.func_code.co_varnames[:narg]
    ranges = {}
    for vname in vnames:
        if 'limit_'+vname in kwd:
            ranges[vname] = kwd['limit_'+vname]
        else:
            ranges[vname]=guessrange

    for i in xrange(ntry):
        arg = []
        for vname in vnames:
            arg.append(randfr(ranges[vname]))
        try:
            thisfom = fom(*arg)
            if first or thisfom<minfom:
                first = False
                minfom = thisfom
                minparam =  arg
        except ZeroDivisionError:
            pass
        
    print minparam,minfom
    ret = {}
    for vname,bestguess in zip(vnames,minparam):
        ret[vname] = bestguess
    if draw:
        fom(*minparam)
        fom.draw()
    return ret

def draw_contour2d(fit,m,var1,var2,bins=12,bound1=None,bound2=None,lh=True):
    x1s,x2s,y = val_contour2d(fit,m,var1,var2,bins=bins,bound1=bound1,bound2=bound2)
    y-=np.min(y)
    v = np.array([0.5,1,1.5])
    if not lh: v*=2
    CS = plt.contour(x1s,x2s,y,v,colors=['b','k','r'])
    plt.clabel(CS)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.axhline(m.values[var2],color='k',ls='--')
    plt.axvline(m.values[var1],color='k',ls='--')
    plt.grid(True)
    return x1s,x2s,y,CS
    
def val_contour2d(fit,m,var1,var2,bins=12,bound1=None,bound2=None):
    assert(var1 != var2)
    if bound1 is None: bound1 = (m.values[var1]-2*m.errors[var1],m.values[var1]+2*m.errors[var1])
    if bound2 is None: bound2 = (m.values[var2]-2*m.errors[var2],m.values[var2]+2*m.errors[var2])
    
    x1s = np.linspace(bound1[0],bound1[1],bins)
    x2s = np.linspace(bound2[0],bound2[1],bins)
    ys = np.zeros((bins,bins))
    
    x1pos = m.var2pos[var1]
    x2pos = m.var2pos[var2]
    
    arg = list(m.args)
    
    for ix1 in xrange(len(x1s)):
        arg[x1pos] = x1s[ix1]
        for ix2 in xrange(len(x2s)):
            arg[x2pos] = x2s[ix2]
            ys[ix1,ix2] = fit(*arg)
    return x1s,x2s,ys

def draw_contour(fit,m,var,bound=None,step=None,lh=True):
    x,y = val_contour(fit,m,var,bound=bound,step=step)
    plt.plot(x,y)
    plt.grid(True)
    plt.xlabel(var)
    if lh: plt.ylabel('-Log likelihood')

    minpos = np.argmin(y)
    ymin = y[minpos]
    tmpy = y-ymin
    #now scan for minpos to the right until greater than one
    up = {True:0.5,False:1.}[lh]
    righty = np.power(tmpy[minpos:] - up,2)
    #print righty
    right_min = np.argmin(righty)
    rightpos = right_min+minpos
    lefty = np.power((tmpy[:minpos] - up),2)
    left_min = np.argmin(lefty)
    leftpos = left_min
    print leftpos,rightpos
    le = x[minpos] - x[leftpos]
    re = x[rightpos] - x[minpos]
    print (le,re)
    vertical_highlight(x[leftpos],x[rightpos])
    plt.figtext(0.5,0.5,'%s = %7.3e ( -%7.3e , +%7.3e)'%(var,x[minpos],le,re),ha='center')
    
def val_contour(fit,m,var,bound=None, step=None):
    arg = list(m.args)
    pos = m.var2pos[var]
    val = m.values[var]
    error = m.errors[var]
    if bound is None: bound = (val-2.*error,val+2.*error)
    if step is None: step=error/100.
    
    xs = np.arange(bound[0],bound[1],step)
    ys = np.zeros(len(xs))
    for i,x in enumerate(xs):
        arg[pos] = x
        ys[i] = fit(*arg)

    return xs,ys
