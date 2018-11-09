from probfit.pdf import exponential
from probfit.plotting import draw_pdf
from collections import OrderedDict
import matplotlib.pyplot as plt

bound = (0, 5)

arg = OrderedDict(tau=0.5)
draw_pdf(exponential, arg=arg, bound=bound, label=str(arg), density=False,
         bins=100)

arg = OrderedDict(tau=1.0)
draw_pdf(exponential, arg=arg, bound=bound, label=str(arg), density=False,
         bins=100)

arg = OrderedDict(tau=1.5)
draw_pdf(exponential, arg=arg, bound=bound, label=str(arg), density=False,
         bins=100)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
