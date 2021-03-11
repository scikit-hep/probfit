# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from probfit.pdf import doublecrystalball
from probfit.plotting import draw_normed_pdf

bound = (-2, 12)

arg = dict(alpha=1.0, alpha2=2.0, n=2.0, n2=4, mean=5, sigma=1)
draw_normed_pdf(doublecrystalball, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(alpha=2, alpha2=1, n=7.0, n2=10.0, mean=5, sigma=1)
draw_normed_pdf(doublecrystalball, arg=arg, bound=bound, label=str(arg), density=True)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
