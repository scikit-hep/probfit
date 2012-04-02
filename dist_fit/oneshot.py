from .cdist_fit import UnbinnedML
from minuit import Minuit, MinuitError
from matplotlib import pyplot as plt
import numpy as np
def randfr(r):
    b = r[1]
    a = r[0]
    return np.random.ranf()*(b-a)+a

def fit_uml(f,data,*arg,**kwd):
    uml = UnbinnedML(f,data)
    m = Minuit(uml,**kwd)
    
    m.printMode=1
    try:
        m.migrad()
    except MinuitError as e:
        plt.figure()
        uml.show()
        print m.values
        raise e
    #m.minos()
    return (uml,m)

def fit_binx2(f,data,*arg,**kwd):
    uml = BinnedChi2(f,data)
    m = Minuit(uml,**kwd)
    
    m.printMode=1
    try:
        m.migrad()
    except MinuitError as e:
        plt.figure()
        uml.show()
        print m.values
        raise e
    #m.minos()
    return (uml,m)

def compare_uml(f,data,*arg,**kwd):
    narg = fom.func_code.co_argcount
    vnames = fom.func_code.co_varnames[:narg]
    arg = [kwd[name] for name in vnames]
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
