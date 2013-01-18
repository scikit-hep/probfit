from probfit.pdf import doublegaussian
from probfit.plotting import draw_normed_pdf
import matplotlib.pyplot as plt

bound = (-10, 10)

arg = dict(mean=1., sigma_L=1, sigma_R=2)
draw_normed_pdf(doublegaussian, arg=arg, bound=bound, label=str(arg))

arg = dict(mean=1., sigma_L=0.5, sigma_R=3)
draw_normed_pdf(doublegaussian, arg=arg, bound=bound, label=str(arg))

plt.grid(True)
plt.legend().get_frame().set_alpha(0.5)
