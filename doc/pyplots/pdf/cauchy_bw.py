from probfit.pdf import cauchy, rtv_breitwigner
from probfit.plotting import draw_pdf
import matplotlib.pyplot as plt

bound = (1, 7.0)

arg = dict(m=5.28, gamma=0.5)
draw_pdf(cauchy, arg=arg, bound=bound, label='cauchy'+str(arg), density=True)

arg = dict(m=5.28, gamma=1.0)
draw_pdf(cauchy, arg=arg, bound=bound, label='cauchy'+str(arg), density=True)

arg = dict(m=5.28, gamma=1.0)
draw_pdf(rtv_breitwigner, arg=arg, bound=bound, label='bw'+str(arg), density=True)

arg = dict(m=5.28, gamma=2.0)
draw_pdf(rtv_breitwigner, arg=arg, bound=bound, label='bw'+str(arg), density=True)

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
