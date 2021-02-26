from probfit.pdf import doublecrystalball
from probfit.plotting import draw_normed_pdf
import matplotlib.pyplot as plt

bound = (-2, 12)

arg = dict(alpha=1.,alpha2=2., n=2.,n2=4, mean=5, sigma=1)
draw_normed_pdf(crystalball, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(alpha=2, alpha2=1, n=7., n2=10., mean=5, sigma=1)
draw_normed_pdf(crystalball, arg=arg, bound=bound, label=str(arg), density=True)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
