# -*- coding: utf-8 -*-
from collections import OrderedDict

import matplotlib.pyplot as plt

from probfit.pdf import ugaussian
from probfit.plotting import draw_pdf

bound = (-10, 10)

arg = OrderedDict(mean=2, sigma=1)
draw_pdf(ugaussian, arg=arg, bound=bound, label=str(arg), density=False)

arg = OrderedDict(mean=-3, sigma=2)
draw_pdf(ugaussian, arg=arg, bound=bound, label=str(arg), density=False)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
