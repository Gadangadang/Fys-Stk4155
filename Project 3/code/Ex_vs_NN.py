from ML_PDE_solver import *
import tensorflow as tf
import ExplicitSolver as ES
import numpy as np
import time

def err_comparison(x, times, solutions, legends, name):

    dx = x[1]-x[0]
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    for i in range(len(solutions)):
        error = calc_MSE(x, times[i], solutions[i])
        plt.plot(times[i], error, label = legends[i])

    # plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$t$", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.ylim(bottom = 1e-20)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.legend(fontsize = 13)
    if name is not None:
        plt.savefig(f"../article/figures/{name}.pdf", bbox_inches="tight")
    plt.show()

def u_exact(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


def calc_MSE(x, t, u):
    MSE = np.zeros(len(t))
    for i, time in enumerate(t):
        MSE[i] = np.mean((u_exact(x, time) - u[i])**2)
    return MSE


if __name__ == "__main__":
    # Common settings
    dx = 0.01
    L = 1
    T = 1


    # Explicit Solver
    I_ex = lambda x: np.sin(np.pi * x)
    dt = 0.5 * dx ** 2
    c = 0
    d = 0
    ESS = ES.ExplicitSolver(I_ex, L, T, dx, dt, 0, 0, False)
    u_ex = ESS.run_simulation()
    t_ex = ESS.t



    # Neural network solver
    tf.random.set_seed(123)
    dt = dx
    lr = 5e-2
    epochs = 2e3
    x = np.linspace(0, L, int(L / dx))
    t_NN = np.linspace(0, T, int(T / dt))

    # Place tensors on the CPU
    with tf.device("/CPU:0"):  # Write '/GPU:0' for large networks
        ML = NeuralNetworkPDE(x, t_NN, int(epochs), I, lr)
        loss = ML.train()
        u_NN_2k = np.array(ML())

        epochs = 2e4
        ML = NeuralNetworkPDE(x, t_NN, int(epochs), I, lr)
        loss = ML.train()
        u_NN_20k = np.array(ML())




    err_comparison(x, [t_ex, t_NN, t_NN], [u_ex, u_NN_2k, u_NN_20k], ["Explicit", "Neural network, 2k epochs", "Neural network, 20k epochs"], name = "error_comparison")
