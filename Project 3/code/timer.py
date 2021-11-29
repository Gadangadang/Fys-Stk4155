from ML_PDE_solver import NeuralNetworkPDE
import tensorflow as tf
import ExplicitSolver as ES
import numpy as np
import time


if __name__ == "__main__":
    I = lambda x: tf.sin(np.pi * x)
    L = 1
    T = 1
    dx = 1 / 100
    dt = 0.5 * dx ** 2
    dt1 = dt * 2 / dx
    x = np.linspace(0, L, int(L / dx))
    t = np.linspace(0, T, int(T / dt1))
    epochs = 150
    lr = 5e-2
    c = 0
    d = 0

    tic = time.perf_counter()
    ESS = ES.ExplicitSolver(I, L, T, dx, dt, c, d)
    solution = ESS.run_simulation()
    toc = time.perf_counter()
    print("Runtime FD: {:.2f}s".format(toc - tic))

    tic1 = time.perf_counter()
    ML = NeuralNetworkPDE(x, t, epochs, I, lr)
    loss = ML.train()
    u_complete = ML()
    toc1 = time.perf_counter()
    print("Runtime ML: {:.2f}s".format(toc1 - tic1))

# Timing results from Mikkels mac
# Finite: 0.12 s
# Network: 4.01 s
