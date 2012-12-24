#decorators
#@normalized_function(xmin,xmax)
#def function_i_have_no_idea_how_to_normalize(x,y,z)
#   return complicated_function(x,y,z)
#
class normalized:
    """

    :param xmin:
    :param xmax:
    """

    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
    def __call__(self,f):
        return Normalized(f,(self.xmin,self.xmax))


class rename_parameters:
    """

    :param arg:
    """

    def __init__(self,*arg):
        self.arg = arg
    def __call__(self,f):
        return FakeFunc(f,self.arg)


def extended(f):
    """

    :param f:
    :return:
    """
    return Extended(f)
