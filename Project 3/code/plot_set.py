import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("bmh")
sns.color_palette("hls", 1)

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def color_cycle(num_color):
    """ get color from matplotlib
        color cycle
        use as: color = color_cycle(3) """
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return color[num_color]

import numpy as np
import statsmodels.api as sm
def lin_fit(x,y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    res = model.fit()
    b, a = res.params
    b_err, a_err = res.bse
    return a, b, a_err, b_err

def decimals(val):
    return  int(np.ceil(-np.log10(val)))

#--- Plot commands ---#
# plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
# plt.xlabel(r"$x$", fontsize=14)
# plt.ylabel(r"$y$", fontsize=14)
# plt.legend(fontsize = 13)
# plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
# plt.savefig("../article/figures/figure.pdf", bbox_inches="tight")
