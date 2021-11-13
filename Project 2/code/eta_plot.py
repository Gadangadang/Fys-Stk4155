import numpy as np
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from matplotlib.ticker import MaxNLocator

def sigmoid_activation(x):
    return 1/(1+np.exp(-x))


epochs = np.linspace(0, 100, 10001)
eta_0 = 1
#dropp_time = 12
#k = 0.5
#A = 1/sigmoid_activation(k*dropp_time)


def eta_func(k, epoch, index):
    return eta_0 * A*sigmoid_activation(k*(dropp_time[index]-epoch))

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

dropp_time = [0, 25, 75, 35, -10]
for index, value in enumerate([0, 0.1, 0.4, 0.4, 0.05]):

    A = 1/sigmoid_activation(value*dropp_time[index])
    plt.plot(epochs, eta_func(value, epochs, index), label=r"k = {:.1f}, $t_D$ = {}".format(value, dropp_time[index]))


plt.xlabel(r"$\xi$", fontsize=14)
plt.ylabel(r"$\eta (k, \xi)$ $[\eta_0]$", fontsize=14)
plt.title(r"$\eta_0$ = {} for different k and $t_D$".format(eta_0) )
plt.legend(fontsize=13)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
plt.savefig(f"../article/figures/eta_plot.pdf",
            bbox_inches="tight")
plt.show()
