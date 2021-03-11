# -*- coding: utf-8 -*-
from collections import OrderedDict

import matplotlib.pyplot as plt

from probfit.pdf import johnsonSU
from probfit.plotting import draw_pdf

bound = (-10, 10)

arg = OrderedDict(mean=2, sigma=1, nu=-4, tau=0.5)
draw_pdf(johnsonSU, arg=arg, bound=bound, label=str(arg), density=False, bins=200)

arg = OrderedDict(mean=-3, sigma=2, nu=+4, tau=0.1)
draw_pdf(johnsonSU, arg=arg, bound=bound, label=str(arg), density=False, bins=200)

arg = OrderedDict(mean=0, sigma=3, nu=+2, tau=0.9)
draw_pdf(johnsonSU, arg=arg, bound=bound, label=str(arg), density=False, bins=200)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
