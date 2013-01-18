from probfit.pdf import cruijff
from probfit.plotting import draw_normed_pdf
import matplotlib.pyplot as plt

bound = (5.22, 5.30)

arg = dict(m_0=5.28, sigma_L=0.005, sigma_R=0.005, alpha_R=0., alpha_L=0.1)
draw_normed_pdf(cruijff, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(m_0=5.28, sigma_L=0.005, sigma_R=0.005, alpha_R=0., alpha_L=0.5)
draw_normed_pdf(cruijff, arg=arg, bound=bound, label=str(arg), density=True)

arg = dict(m_0=5.28, sigma_L=0.002, sigma_R=0.005, alpha_R=0., alpha_L=0.01)
draw_normed_pdf(cruijff, arg=arg, bound=bound, label=str(arg), density=True)

plt.grid(True)
plt.legend(loc='upper left', prop={'size': 8}).get_frame().set_alpha(0.5)
