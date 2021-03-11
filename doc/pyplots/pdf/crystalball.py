# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from probfit.pdf import crystalball
from probfit.plotting import draw_normed_pdf

bound = (5.22, 5.30)

arg = dict(alpha=1.0, n=2.0, mean=5.28, sigma=0.01)
draw_normed_pdf(crystalball, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(alpha=0.5, n=10.0, mean=5.28, sigma=0.005)
draw_normed_pdf(crystalball, arg=arg, bound=bound, label=str(arg), density=True)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
