from iminuit.util import describe #use iminuit describe..


def parse_arg(f, kwd, offset=0):
    """
    convert dictionary of keyword argument and value to positional argument
    equivalent to::

        vnames = describe(f)
        return tuple([kwd[k] for k in vnames[offset:]])

    """
    vnames = describe(f)
    return tuple([kwd[k] for k in vnames[offset:]])


def remove_prefix(s, prefix):
    if prefix is None:
        return s
    if s.startswith(prefix+'_'):
        l = len(prefix)+1
        return s[l:]
    else:
        return s
