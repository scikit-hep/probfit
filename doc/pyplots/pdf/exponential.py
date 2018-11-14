from probfit.pdf import exponential
from probfit.plotting import draw_pdf
import matplotlib.pyplot as plt

bound = (0, 5)

arg = {"lambda": 0.5}
draw_pdf(exponential, arg=arg, bound=bound, label=str(arg), density=False,
         bins=100)

arg = {"lambda": 1.0}
draw_pdf(exponential, arg=arg, bound=bound, label=str(arg), density=False,
         bins=100)

arg = {"lambda": 1.5}
draw_pdf(exponential, arg=arg, bound=bound, label=str(arg), density=False,
         bins=100)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
