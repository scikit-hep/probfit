How do I .....?
===============

Tell probfit to use my analytical integral
------------------------------------------
probfit checks for method call integrate(bound, nint, *arg).
If such method is available it calls that method to compute definite integral.
If not it falls back to simpson3/8(cubic approximation).
::
    from probfit import integrate1d
    def line(x, m, c):
        return m*x+c

    # compute integral of line from x=(0,1) using 10 intevals with m=1. and c=2.
    # all probfit internal use this
    
    # no integrate method available probfit use simpson3/8
    print integrate1d(line, (0,1), 10, (1.,2.) ) 

    # Let us illustrate the point by forcing it to have integral that's off by
    # factor of two
    def wrong_line_integral(bound, nint, m, c):
        a, b = bound
        return 2*(m*(b**2/2.-a**2/2.)+c*(b-a)) # I know this is wrong 
    
    line.integrate = wrong_line_integral
    # line.integrate = lambda bound, nint, m, c: blah blah # this works too
    
    # yes off by factor of two
    print integrate1d(line, (0,1), 10, (1.,2.))
