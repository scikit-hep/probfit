# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from probfit.pdf import argus
from probfit.plotting import draw_normed_pdf

bound = (5.22, 5.30)

arg = dict(c=5.29, chi=1.0, p=0.5)
draw_normed_pdf(argus, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(c=5.29, chi=1.0, p=0.4)
draw_normed_pdf(argus, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(c=5.29, chi=2.0, p=0.5)
draw_normed_pdf(argus, arg=arg, bound=bound, label=str(arg), density=True)


plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
