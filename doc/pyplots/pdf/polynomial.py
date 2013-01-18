from probfit.pdf import Polynomial
from probfit.plotting import draw_pdf
import matplotlib.pyplot as plt
bound = (-10, 10)

p = Polynomial(3)
arg = dict(c_0=0., c_1=1, c_2=2, c_3=3)
draw_pdf(p, arg=arg, bound=bound, label=str(arg), density=False)

p = Polynomial(2)
arg = dict(c_0=0., c_1=1, c_2=2)
draw_pdf(p, arg=arg, bound=bound, label=str(arg), density=False)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
