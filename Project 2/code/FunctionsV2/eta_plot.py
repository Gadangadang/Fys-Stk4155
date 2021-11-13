import numpy as np
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
from import_folders import *
import_all_folders()
from NeuralNetwork import NeuralNetwork

def sigmoid_activation(x):
    return 1/(1+np.exp(-x))


epochs = np.arange(0, 21)
eta_0 = 0.1
dropp_time=5
A = 1
k = 0.5

def eta_func(k, epoch):
    return eta_0 * A*sigmoid_activation(k*(dropp_time-epoch))


eta_vals = eta_func(k, epochs)
print(eta_vals)
#plt.plot(epochs, eta_vals)
#plt.show()

print(np.random.randn(2,2))
