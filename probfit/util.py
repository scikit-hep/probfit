from iminuit.util import describe #use iminuit describe..

def parse_arg(f,kwd,offset=0):
    """

    :param f:
    :param kwd:
    """
    vnames = describe(f)
    return tuple([kwd[k] for k in vnames[offset:]])


def safe_getattr(o,attr,default=None):
    if hasattr(o,attr):
        return getattr(o,attr)
    else:
        return default