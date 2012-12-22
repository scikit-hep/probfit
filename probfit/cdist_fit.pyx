#cython: embedsignature=True
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp,pow,fabs,log,tgamma,lgamma,log1p,sqrt
from matplotlib import pyplot as plt
from .common import *
from cython.parallel import prange, parallel, threadid
import multiprocessing as mp
from warnings import warn
from cdist_func cimport *
cdef extern from "math.h":
    bint isnan(double x)


cdef double compute_bin_lh_f(f, 
                    np.ndarray[np.double_t] edges,
                    np.ndarray[np.double_t] h, #histogram,
                    np.ndarray[np.double_t] w2,
                    double N, #sum of h
                    tuple arg, double badvalue, 
                    bint extend,bint use_sumw2) except *:
    cdef int i
    cdef int n = len(edges)

    cdef np.ndarray[np.double_t] fedges = cvectorize_f(f,edges,arg)
    cdef np.ndarray[np.double_t] midvalues = (fedges[1:]+fedges[:-1])/2
    cdef double ret = 0.
    cdef double bw = 0.
    cdef double E = cintegrate1d(f,(edges[0],edges[-1]),10000,arg)
    cdef double factor=0.
    cdef double th=0.
    cdef double tw=0.
    cdef double tm=0.
    for i in range(n-1):#h has length of n-1
        #ret -= h[i]*log(midvalues[i])#non zero subtraction
        bw = edges[i+1]-edges[i]
        th = h[i]
        tm = midvalues[i]
        if not extend:
            if not use_sumw2:
                ret -= cxlogyx(th,tm*N*bw)+(th-tm*bw*N)#h[i]*log(midvalues[i]/nh[i]) #subtracting h[i]*log(h[i]/(N*bw))
            else:
                if w2[i]<1e-200: continue
                tw = w2[i]
                tw = sqrt(tw)
                factor = th/tw
                ret -= factor*(cwlogyx(th,tm*N*bw,th)+(th-tm*bw*N))
        else:
            #print 'h',h[i],'midvalues',midvalues[i]*bw
            if not use_sumw2:
                ret -= cxlogyx(th,tm*bw)+(th-tm*bw)
            else:
                if w2[i]<1e-200: continue
                tw = w2[i]
                tw = sqrt(tw)
                factor = th/tw
                ret -= factor*(cwlogyx(th,tm*bw,th)+(th-tm*bw))
    return ret


cdef double cgauss(double x, double mean, double sigma):
    return 1/sqrt(2*pi*sigma)*exp((x-mean)*(x-mean)/sigma/sigma/2.);


cdef np.ndarray[np.double_t] midvalues(f,np.ndarray[np.double_t]edges,tuple arg):
    cdef int n = len(edges)
    cdef int i
    cdef np.ndarray[np.double_t] ret=np.zeros(n-1)
    cdef np.ndarray[np.double_t] tmp=np.zeros(n)
    for i in range(n):
        tmp[i] = f(edges[i],*arg)
    for i in range(n-1):
        ret[i] = (tmp[i]+tmp[i+1])/2
    return ret


cdef double compute_nll(f,np.ndarray data,w,arg,double badvalue) except *:
    cdef int i=0
    cdef double lh=0
    cdef double nll=0
    cdef double ret=0
    cdef double thisdata=0
    cdef np.ndarray[np.double_t] data_ = data
    cdef int data_len = len(data)
    cdef np.ndarray[np.double_t] w_
    if w is None:
        for i in range(data_len):
        #for i in prange(data_len,nogil=True): #nope!!!! gilllllllll
            thisdata = data_[i]
            lh = f(thisdata,*arg)
            if lh<=0:
                ret = badvalue
                break
            else:
                ret+=log(lh)
    else:
        w_ = w
        for i in range(data_len):
        #for i in prange(data_len,nogil=True): #nope!!!! gillllllll
            thisdata = data_[i]
            lh = f(thisdata,*arg,nogil=True)
            if lh<=0:
                ret = badvalue
                break
            else:
                ret+=log(lh)*w_[i]        
    return -1*ret


cdef double compute_chi2_f(f,np.ndarray[np.double_t] x,np.ndarray[np.double_t] y ,
                np.ndarray[np.double_t]error,np.ndarray[np.double_t]weights,tuple arg) except *:
    cdef int usew = 1 if weights is not None else 0
    cdef int usee = 1 if error is not None else 0
    cdef int i
    cdef int datalen = len(x)
    cdef double diff
    cdef double fx
    cdef double ret=0
    cdef double err
    for i in range(datalen):
        fx = f(x[i],*arg)
        diff = fx-y[i]
        if usee==1:
            err = error[i]
            if err<1e-10:
                raise ValueError('error contains value too small or negative')
            diff = diff/error[i]
        diff *= diff
        if usew==1:
            diff*=weights[i]
        ret += diff
    return ret


cdef double compute_bin_chi2_f(f,
                np.ndarray[np.double_t] x,np.ndarray[np.double_t] y,
                np.ndarray[np.double_t]error,np.ndarray[np.double_t] binwidth,
                np.ndarray[np.double_t]weights,tuple arg) except *:
    cdef int usew = 1 if weights is not None else 0
    cdef int usee = 1 if error is not None else 0
    cdef int i
    cdef int datalen = len(x)
    cdef double diff
    cdef double fx
    cdef double ret=0
    cdef double err
    cdef double bw
    for i in range(datalen):
        bw = binwidth[i]
        fx = f(x[i],*arg)*bw
        diff = fx-y[i]
        if usee==1:
            err = error[i]
            if err<1e-10:
                raise ValueError('error contains value too small or negative')
            diff = diff/error[i]
        diff *= diff
        if usew==1:
            diff*=weights[i]
        ret += diff
    return ret


def compute_cdf(np.ndarray[np.double_t] pdf, np.ndarray[np.double_t] x) :

    cdef int i
    cdef int n = len(pdf)
    cdef double lpdf
    cdef double rpdf
    cdef bw
    cdef np.ndarray[np.double_t] ret
    ret = np.zeros(n)
    ret[0] = 0
    for i in range(1,n):#do a trapezoid sum
        lpdf = pdf[i]
        rpdf = pdf[i-1]
        bw = x[i]-x[i-1]
        ret[i] = 0.5*(lpdf+rpdf)*bw + ret[i-1]
    return ret

#invert cdf useful for making toys
def invert_cdf(np.ndarray[np.double_t] r, np.ndarray[np.double_t] cdf, np.ndarray[np.double_t] x):
    cdef np.ndarray[np.int_t] loc = np.searchsorted(cdf,r,'right')
    cdef int n = len(r)
    cdef np.ndarray[np.double_t] ret = np.zeros(n)
    cdef int i = 0
    cdef int ind
    cdef double ly
    cdef double lx
    cdef double ry
    cdef double rx
    cdef double minv
    for i in range(n):
        ind = loc[i]
        #print ind,i,len(loc),len(x)
        ly = cdf[ind-1]
        ry = cdf[ind]
        lx = x[ind-1]
        rx = x[ind]
        minv = (rx-lx)/(ry-ly)
        ret[i] = minv*(r[i]-ly)+lx
    return ret


cdef double compute_chi2(np.ndarray[np.double_t] actual, np.ndarray[np.double_t] expected, np.ndarray[np.double_t] err):
    cdef int i=0
    cdef int maxi = len(actual)
    cdef double a
    cdef double e
    cdef double er
    cdef double ret
    for i in range(maxi):
        e = expected[i]
        a = actual[i]
        er = err[i]
        if er<1e-10:
            raise ValueError('error contains value too small or negative')
        ea = (e-a)/er
        ea *=ea
        ret +=ea
    return ret


cdef class UnbinnedMLWorker:
    cdef np.ndarray data
    cdef public object f
    cdef object weights
    cdef double badvalue
    def __init__(self,f,data,weights,badvalue):
        self.f=f
        self.data=data
        self.weights=weights
        self.badvalue=badvalue
    def __call__(self,*arg):
        return compute_nll(self.f,self.data,self.weights,arg,self.badvalue)


cdef class UnbinnedMLP:
    """
    This will die on throwing exception don't use this ever I should change this to zmq instead
    """
    cdef public object f
    cdef object weights
    cdef public object func_code
    cdef np.ndarray data
    cdef int data_len
    cdef double badvalue
    cdef tuple last_arg
    cdef list data_chunk
    cdef list weight_chunk
    #cdef object pool
    cdef int num_chunk
    cdef result_tmp
    def __init__(self, f, data ,weights=None,badvalue=-100000):
        #self.vf = np.vectorize(f)
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.weights = weights
        #only make copy when type mismatch
        self.data = float2double(data)
        self.data_len = len(data)
        if weights is None:
            self.weights = np.zeros(self.data_len)
            self.weights.fill(1.)
        self.badvalue = badvalue
        numcpu = mp.cpu_count()
        numworker = numcpu
        
        #self.pool = mp.Pool(numworker)
        self.num_chunk = numworker
        dic = int(self.data_len/numworker)
        self.data_chunk = []
        self.weight_chunk = []
        for i in xrange(numworker):
            if i!=numworker:
                self.data_chunk.append(self.data[i*dic:(i+1)*dic])
                self.weight_chunk.append(self.weights[i*dic:(i+1)*dic])
            else:
                self.data_chunk.append(self.data[i*dic:])
                self.weight_chunk.append(self.weights[i*dic:])
        self.result_tmp = []*self.num_chunk
        
    def __call__(self,*arg):
        self.last_arg = arg
        
        jobs = []
        for i in range(self.num_chunk):
            p = mp.Process(target=self.call_worker,args=(i,arg,))
            jobs.append(p)
            p.start()
        
        for p in jobs:
            p.join()
        
        return sum(nll)
    
    def call_worker(self,i,arg):
        print 'call '+str(i)
        ret = compute_nll(self.f,self.data_chunk[i],self.weight_chunk[i],arg,self.badvalue)
        print 'ready to return '+str(i)+' result ='+str(ret)
        self.result_tmp[i]=ret
        return ret
    
    def draw(self,minuit=None,bins=100,ax=None,range=None,parmloc=(0.05,0.95),nfbins=500):
        if ax is None: ax=plt.gca()

        n,e,patches = ax.hist(self.data,bins=bins,weights=self.weights,
            histtype='step',range=range,normed=True)
        m = mid(e)
        vf = np.vectorize(self.f)
        fxs = np.linspace(e[0],e[-1],nfbins)
        # v = vf(fxs,*self.last_arg)
        # plt.plot(fxs,v,color='r')
        v = vf(fxs,*self.last_arg)
        ax.plot(fxs,v,color='r')

        ax.grid(True)
        minu = minuit
        ax = plt.gca()
        if minu is not None:
            #build text
            txt = u'';
            for k,v  in minu.values.items():
                err = minu.errors[k]
                txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
            print txt
            ax.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)        


cdef class UnbinnedML:
    cdef public object f
    cdef object weights
    cdef public object func_code
    cdef np.ndarray data
    cdef int data_len
    cdef double badvalue
    cdef tuple last_arg
    cdef object pool
    
    def __init__(self, f, data ,weights=None,badvalue=-100000):
        #self.vf = np.vectorize(f)
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.weights = weights
        #only make copy when type mismatch
        self.data = float2double(data)
        self.data_len = len(data)
        self.badvalue = badvalue

    def __call__(self,*arg):
        self.last_arg = arg
        return compute_nll(self.f,self.data,self.weights,arg,self.badvalue)
    
    @cython.binding(True)
    def draw(self,minuit=None,bins=100,ax=None,range=None,parmloc=(0.05,0.95),nfbins=500,print_par=False):
        if ax is None: ax=plt.gca()
        arg = self.last_arg
        if minuit is not None: arg = minuit.args
        n,e,patches = ax.hist(self.data,bins=bins,weights=self.weights,
            histtype='step',range=range,normed=True)
        m = mid(e)
        vf = np.vectorize(self.f)
        fxs = np.linspace(e[0],e[-1],nfbins)
        # v = vf(fxs,*self.last_arg)
        # plt.plot(fxs,v,color='r')
        v = vf(fxs,*arg)
        ax.plot(fxs,v,color='r')
        
        ax.grid(True)
        minu = minuit
        ax = plt.gca()
        if minu is not None:
            #build text
            txt = u'';
            for k,v  in minu.values.items():
                err = minu.errors[k]
                txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
            if print_par: print txt
            ax.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)
                
    def show(self,*arg,**kwd):
        self.draw(*arg,**kwd)
        plt.show()


#fit a line with given function using minimizing chi2
cdef class Chi2Regression:
    cdef public object f
    cdef object weights
    cdef object error
    cdef public object func_code
    cdef int data_len
    cdef double badvalue
    cdef int ndof
    cdef np.ndarray x
    cdef np.ndarray y
    cdef tuple last_arg
    
    def __init__(self, f, x, y,error=None,weights=None,badvalue=1000000):
        #self.vf = np.vectorize(f)
        self.f = f
        self.func_code = FakeFuncCode(f,dock=True)
        self.weights = float2double(weights) 
        self.error = float2double(error)
        self.x = float2double(x)
        self.y = float2double(y)
        self.data_len = len(x)
        self.badvalue = badvalue
        self.ndof = self.data_len - (self.func_code.co_argcount-1)
        
    def __call__(self,*arg):
        self.last_arg = arg
        return compute_chi2_f(self.f,self.x,self.y,self.error,self.weights,arg)

    def draw(self,minuit=None,parmloc=(0.05,0.95),print_par=False):
        arg = self.last_arg
        if minuit is not None: arg = minuit.args
        vf = np.vectorize(self.f)
        x=self.x
        y=self.y
        err = self.error
        expy = vf(x,*arg)
        
        if err is None:
            plt.plot(x,y,'+')
        else:
            plt.errorbar(x,y,err,fmt='.')
        plt.plot(x,expy,'r-')
        ax = plt.gca()
        minu = minuit
        if minu is not None:
            #build text
            txt = u'';
            for k,v  in minu.values.items():
                err = minu.errors[k]
                txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
            if print_par: print txt
            chi2 = self(*self.last_arg)
            txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2/self.ndof,chi2,self.ndof)
            plt.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)
  
    def show(self,*arg):
        self.draw(*arg)
        plt.show()


cdef class BinnedChi2:
    cdef public object f
    cdef public object vf
    cdef public object func_code
    cdef np.ndarray h
    cdef np.ndarray err
    cdef np.ndarray edges
    cdef np.ndarray midpoints
    cdef np.ndarray binwidth
    cdef int bins
    cdef double mymin
    cdef double mymax
    cdef double badvalue
    cdef tuple last_arg
    cdef int ndof
    def __init__(self, f, data, bins=40, weights=None,range=None, sumw2=False,badvalue=1000000):
        self.f = f
        self.vf = np.vectorize(f)
        self.func_code = FakeFuncCode(f,dock=True)
        if range is None:
            range = minmax(data)
        self.mymin,self.mymax = range 
        
        h,self.edges = np.histogram(data,bins,range=range,weights=weights)
        self.h = float2double(h)
        self.midpoints = mid(self.edges)
        self.binwidth = np.diff(self.edges)
        #sumw2 if requested
        if weights is not None and sumw2:
            w2 = weights*weights
            sw2,_ = np.histogram(data,bins,range=range,weights=w2)
            self.err = np.sqrt(sw2)
        else:
            self.err = np.sqrt(self.h)
        #check if error is too small
        if np.any(self.err<1e-5):
            raise ValueError('some bins are too small to do a chi2 fit. change your range')
        self.bins = bins
        self.badvalue = badvalue
        self.ndof = self.bins-(self.func_code.co_argcount-1)#fix this taking care of fixed parameter
    
    #lazy mid point implementation
    def __call__(self,*arg):
        self.last_arg = arg
        return compute_bin_chi2_f(self.f,self.midpoints,self.h,self.err,self.binwidth,None,arg)
    
    # def __call__(self,*arg):
    #        #can be optimized much better than this
    #        cdef np.ndarray[np.double_t] edges_values
    #        cdef np.ndarray[np.double_t] expy
    #        self.last_arg = arg
    #        edges_values = self.vf(self.edges,*arg)
    #        expy = mid(edges_values)
    #        return compute_chi2(self.h,expy,self.err)

    def draw(self,minuit=None,parmloc=(0.05,0.95),fbins=1000,ax = None,print_par=False):
        if ax is None: ax = plt.gca()
        arg = self.last_arg
        if minuit is not None: arg = minuit.args
        m = mid(self.edges)
        ax.errorbar(m,self.h,self.err,fmt='.')
        #assume equal spacing
        #self.edges[0],self.edges[-1]
        bw = self.edges[1]-self.edges[0]
        xs = np.linspace(self.edges[0],self.edges[-1],fbins)
        #bw = np.diff(xs)
        xs = mid(xs)
        expy = self.vf(xs,*arg)*bw
        
        ax.plot(xs,expy,'r-')
        
        minu = minuit
        ax.grid(True)

        if minu is not None:
            #build text
            txt = u'';
            for k,v  in minu.values.items():
                err = minu.errors[k]
                txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
            chi2 = self(*self.last_arg)
            txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2/self.ndof,chi2,self.ndof)
            if print_par: print txt
            ax.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)
    
    def show(self,*arg,**kwd):
        self.draw(*arg,**kwd)
        plt.show()


cdef class BinnedLH:
    cdef public object f
    cdef public object vf
    cdef public object func_code
    cdef np.ndarray h
    cdef np.ndarray w
    cdef np.ndarray w2
    cdef double N
    cdef np.ndarray edges
    cdef np.ndarray midpoints
    cdef np.ndarray binwidth
    cdef int bins
    cdef double mymin
    cdef double mymax
    cdef double badvalue
    cdef tuple last_arg
    cdef int ndof
    cdef bint extended
    cdef bint use_w2
    def __init__(self, f, data, bins=40, weights=None, range=None, badvalue=1000000, 
            extended=False, use_w2=False,use_normw=False):
        self.f = f
        self.vf = np.vectorize(f)
        self.func_code = FakeFuncCode(f,dock=True)
        self.use_w2 = use_w2
        self.extended = extended
        
        if range is None: range = minmax(data)
        self.mymin,self.mymax = range 
        self.w = float2double(weights)
        if use_normw: self.w=self.w/np.sum(self.w)*len(self.w)
        h,self.edges = np.histogram(data,bins,range=range,weights=weights)
        self.h = float2double(h)
        self.N = csum(self.h)

        if weights is not None:
            self.w2,_ = np.histogram(data,bins,range=range,weights=weights*weights)
        else:
            self.w2,_ = np.histogram(data,bins,range=range,weights=None)
        self.w2 = float2double(self.w2)
        self.midpoints = mid(self.edges)
        self.binwidth = np.diff(self.edges)
        
        self.bins = bins
        self.badvalue = badvalue
        self.ndof = self.bins-(self.func_code.co_argcount-1)

    #lazy mid point implementation
    def __call__(self,*arg):
        self.last_arg = arg
        ret = compute_bin_lh_f(self.f, 
                                self.edges,
                                self.h, #histogram,
                                self.w2,
                                self.N, #sum of h
                                arg, self.badvalue, 
                                self.extended, self.use_w2)
        return ret
            
    def draw(self,minuit=None,parmloc=(0.05,0.95),fbins=1000,ax = None,print_par=False):
        if ax is None: ax = plt.gca()
        arg = self.last_arg
        if minuit is not None: arg = minuit.args
        m = mid(self.edges)
        if self.use_w2:
            err = np.sqrt(self.w2)
        else:
            err = np.sqrt(self.h)

        if self.extended:
            ax.errorbar(m,self.h,err,fmt='.')
        else:
            scale = sum(self.h)
            ax.errorbar(m,self.h/scale,err/scale,fmt='.')
        
        #assume equal spacing
        #self.edges[0],self.edges[-1]
        bw = self.edges[1]-self.edges[0]
        xs = np.linspace(self.edges[0],self.edges[-1],fbins)
        #bw = np.diff(xs)
        xs = mid(xs)
        expy = self.vf(xs,*arg)*bw
        #if not self.extended: expy/=sum(expy)
        ax.plot(xs,expy,'r-')

        minu = minuit
        ax.grid(True)

        if minu is not None:
            #build text
            txt = u'';
            sortk = minu.values.keys()
            sortk.sort()
            #for k,v  in minu.values.items():
            val = minu.values
            for k in sortk:
                v = val[k]
                err = minu.errors[k]
                txt += u'%s = %5.4g±%5.4g\n'%(k,v,err)
            #chi2 = self(*self.last_arg)
            #txt+=u'chi2/ndof = %5.4g(%5.4g/%d)'%(chi2,chi2*self.ndof,self.ndof)
            if print_par: print txt
            ax.text(parmloc[0],parmloc[1],txt,ha='left',va='top',transform=ax.transAxes)

    def show(self,*arg,**kwd):
        self.draw(*arg,**kwd)
        plt.show()

