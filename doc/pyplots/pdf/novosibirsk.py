from probfit.pdf import novosibirsk
from probfit.plotting import draw_normed_pdf
import matplotlib.pyplot as plt

bound = (5.22, 5.30)

arg = dict(width=0.005, peak=5.28, tail=0.2)
draw_normed_pdf(novosibirsk, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(width=0.002, peak=5.28, tail=0.2)
draw_normed_pdf(novosibirsk, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(width=0.005, peak=5.28, tail=0.1)
draw_normed_pdf(novosibirsk, arg=arg, bound=bound, label=str(arg), density=True)

plt.grid(True)
plt.legend(loc='upper left').get_frame().set_alpha(0.5)
