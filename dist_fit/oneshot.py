from .cdist_fit import UnbinnedML, BinnedChi2,compute_cdf,invert_cdf
from minuit import Minuit, MinuitError
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as npr
import itertools as itt
import collections
def randfr(r):
    b = r[1]
    a = r[0]
    return np.random.ranf()*(b-a)+a

def fit_uml(f,data,quiet=False,*arg,**kwd):
    uml = UnbinnedML(f,data)
    m = Minuit(uml,**kwd)
    m.up=0.5
    m.strategy=2
    m.printMode=1
    try:
        m.migrad()
    except MinuitError as e:
        if not quiet:
            plt.figure()
            uml.show()
            print m.values
        raise e
    #m.minos()
    return (uml,m)

def fit_binx2(f,data,bins=30, quiet=False,*arg,**kwd):
    uml = BinnedChi2(f,data,bins=bins)
    m = Minuit(uml,**kwd)
    m.strategy=2
    m.printMode=1
    try:
        m.migrad()
    except MinuitError as e:
        if not quiet:
            plt.figure()
            uml.show()
            print m.values
        raise e
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
    
def try_chi2(f,data,bins=40,fbins=1000,*arg,**kwd):
    fom = BinnedChi2(f,data)
    narg = f.func_code.co_argcount
    vnames = f.func_code.co_varnames[1:narg]
    my_arg = [ tuplize(kwd[name]) for name in vnames ]
    h,e,_ = plt.hist(data,bins=bins,histtype='step')
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
    This is very bad at the moment don't use if it's not neccessary
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
