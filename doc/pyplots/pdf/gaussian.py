from probfit.pdf import gaussian
from probfit.plotting import draw_pdf
import matplotlib.pyplot as plt

bound = (-10, 10)

arg = dict(mean=2, sigma=1)
draw_pdf(gaussian, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(mean=-3, sigma=2)
draw_pdf(gaussian, arg=arg, bound=bound, label=str(arg), density=True)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
