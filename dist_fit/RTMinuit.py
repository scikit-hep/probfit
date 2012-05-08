import ROOT
from inspect import getargspec
import sys
from array import array
from common import FCN
class Minuit:
    
    def __init__(self,f,**kwds):
        self.fcn = FCN(f)
        
        args,_,_,_ = getargspec(f)
        narg = len(args)
        
        #maintain 2 dictionary 1 is position to varname
        #and varname to position
        self.args = args
        self.pos2var = {i:k for i,k in enumerate(args)}
        self.var2pos = {k:i for i,k in enumerate(args)}

        self.tmin = ROOT.TMinuit(narg)
        self.prepare(**kwds)
    
    def prepare(self,**kwds):
        self.tmin.SetFCN(self.fcn)

        for i,varname in self.pos2var.items():
            print '---------varname',varname
            initialvalue = kwds[varname] if varname in kwds else 0.
            initialstep = kwds['error_'+varname] if 'error_'+varname in kwds else 0.1
            lrange,urange = kwds['limit_'+varname] if 'limit_'+varname in kwds else (0.,0.)
            ierflg = self.tmin.DefineParameter(i,varname,initialvalue,initialstep,lrange,urange)
            assert(ierflg==0)
        #now fix parameter
        for i,varname in self.pos2var.items():
            if 'fix_'+varname in kwds: self.tmin.FixParameter(i)
        
    def set_up(self,up):
        return self.tmin.SetErrorDef(up)

    def migrad(self):
        return self.tmin.Migrad()

    def minos(self):
        self.tmin.mnmnos

    def values(self):
        ret = {}
        for i,varname in self.pos2var.items():
            tmp_val = ROOT.Double(0.)
            tmp_err = ROOT.Double(0.)
            self.tmin.GetParameter(i,tmp_val,tmp_err)
            print tmp_val
            ret[varname] = float(tmp_val)
        return ret

    def errors(self):
        ret = {}
        for i,varname in self.pos2var.items():
            tmp_val = ROOT.Double(0.)
            tmp_err = ROOT.Double(0.)
            self.tmin.GetParameter(i,tmp_val,tmp_err)
            ret[varname] = float(tmp_err)
        return ret    
            
    def args(self):
        val = self.values
        tmp = []
        for arg in self.args:
            tmp.append(val[arg])
        return tuple(tmp)
        
def main():
    def test(x,y):
        print '*****x,y',x,y
        return (x-2)**2 + (y-3)**2 + 1.
    m = Minuit(test,x=2.1,y=2.9)
    m.migrad()
    print m.values()
    print m.errors()
   
if __name__ == '__main__':
    main()