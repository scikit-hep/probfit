from iminuit.util import describe #use iminuit describe..

def parse_arg(f,kwd,offset=0):
    """
    convert dictionary of keyword argument and value to positional argument
    equivalent to::

        vnames = describe(f)
        return tuple([kwd[k] for k in vnames[offset:]])

    """
    vnames = describe(f)
    return tuple([kwd[k] for k in vnames[offset:]])


def safe_getattr(o,attr,default=None):
    if hasattr(o,attr):
        return getattr(o,attr)
    else:
        return default