#ifdef _MSC_VER //visual c++ doesn't have log1p
    __inline double log1p(double x){
        //From ROOT implementation
        volatile double y = 1 + x;
        return log(y) - ((y-1)-x)/y;
    }
#else
    #include <math.h>
#endif

