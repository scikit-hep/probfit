from .cdist_fit import UnbinnedML
from minuit import Minuit

def fit_uml(f,data,*arg,**keyword):
    uml = UnbinnedML(f,data)
    m = minuit.Minuit(uml,**kwd)
    m.migrad()
    m.minos()
    return (uml,m)
