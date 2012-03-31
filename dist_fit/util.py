import cdist_fit as cdist
import numpy as np
#fake copying func_code with renaming parameters
#and docking off parameters from the front
class FakeFuncCode:
    def __init__(self,f,prmt=None,dock=0):
        #f can either be tuple or function object
        if hasattr(f,'func_code'):#copy function code
            for attr in dir(f.func_code):
                if '__' not in attr: 
                    #print attr, type(getattr(f.func_code,attr))
                    setattr(self,attr,getattr(f.func_code,attr))
            self.co_argcount-=dock
            self.co_varnames = self.co_varnames[dock:]
            if prmt is not None:#rename parameters from the front
                for i,p in enumerate(prmt):
                    self.co_varnames[i] = p
        else:#build a really fake one from bare bone
            raise TypeError('f does not have func_code')

#and a decorator so people can do
#@normalized_function(xmin,xmax)
#def function_i_have_no_idea_how_to_normalize(x,y,z)
#   return complicated_function(x,y,z)
#
class normalized_function:
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
    def __call__(self,f):
        return cdist.Normalize(f,(self.xmin,self.xmax))

class rename_parameters:
    def __init__(self,*arg):
        self.arg = arg
    def __call__(self,f):
        return cdist.FakeFunc(f,self.arg)