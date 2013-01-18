from probfit.pdf import cauchy, rtv_breitwigner
from probfit.plotting import draw_pdf
from collections import OrderedDict
import matplotlib.pyplot as plt

bound = (5.24, 5.32)

arg = OrderedDict(m=5.28, gamma=1)
draw_pdf(cauchy, arg=arg, bound=bound, label='cauchy'+str(arg), density=True)

arg = OrderedDict(m=-5.28, gamma=2)
draw_pdf(cauchy, arg=arg, bound=bound, label='cauchy'+str(arg), density=True)

arg = OrderedDict(m=5.28, gamma=1)
draw_pdf(rtv_breitwigner, arg=arg, bound=bound, label='bw'+str(arg), density=True)

arg = OrderedDict(m=-5.28, gamma=2)
draw_pdf(rtv_breitwigner, arg=arg, bound=bound, label='bw'+str(arg), density=True)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
