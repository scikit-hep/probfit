# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from probfit.pdf import exponential
from probfit.plotting import draw_pdf

_arg = {"lambda": 0.5}
draw_pdf(exponential, arg=_arg, bound=(0, 5), label=str(_arg), density=False, bins=100)

_arg = {"lambda": 1.0}
draw_pdf(exponential, arg=_arg, bound=(0, 5), label=str(_arg), density=False, bins=100)

_arg = {"lambda": 1.5}
draw_pdf(exponential, arg=_arg, bound=(0, 5), label=str(_arg), density=False, bins=100)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
