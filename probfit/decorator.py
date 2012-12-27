#decorators
#@normalized_function(xmin,xmax)
#def function_i_have_no_idea_how_to_normalize(x,y,z)
#   return complicated_function(x,y,z)
#
from .functor import Normalized, Extended
class normalized:
    """
    Normalized decorator

    **Arguments**
        - **bound** normalized bound
        - **nint** option number of integral pieces. Default 1000.

    .. seealso::

        :class:`probfit.functor.Normalized`

    """

    def __init__(self, bound, nint=1000):
        self.bound  = bound
        self.nint = nint

    def __call__(self,f):
        return Normalized(f, self.bound, self.nint)


class extended:
    """
    Extended decorator

    **Arguments**
        - **extname** extended parameter name. Default 'N'

    .. seealso::

        :class:`probfit.functor.Extended`

    """
    def __init__(self, extname='N'):
        self.extname = extname

    def __call__(self,f):
        return Extended(f, extname=self.extname)
