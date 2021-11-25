import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
import matplotlib.animation as animation
import seaborn as sns


class PDE_solver:
    def __init__(self, I, L, T, dx, dt, c, d):
        # Read class arguments
        self.I = I
        self.L = L
        self.T = T
        self.n = 0 # set current timestep = 0
        self.c = c #boundary points x = 0
        self.d = d #boundary points x = Lx

        # Mesh points & dx, dy, dt
        self.Nt = int(T/dt)
        self.Nx = int(L/dx)
        self.dt = dt
        self.dx = dx
        self.C = dt/dx**2

        self.x = np.linspace(0, self.L, self.Nx)

        # Solution array (with space for ghost points)
        self.u = np.zeros((self.Nx))   # solution at t
        self.u_1 = np.zeros((self.Nx)) # solution at t - dt

        # Initial conditions
        self.u_1[1:-1] = I(self.x[1:-1])
        self.u_1[0] = c
        self.u_1[-1] = d


    def boundary_points(self):
        """
        Initialize boundary points.
        """
        self.u[0] = self.c
        self.u[-1] = self.d

    def advance_solution(self):
        """
        Advance solution to get u = u[n+1].
        """
        self.u[1:-1] = self.C*(self.u_1[2:]-2*self.u_1[1:-1] + self.u_1[:-2]) + self.u_1[1:-1]
        self.boundary_points()
        self.n += 1 # update timestep
        self.u, self.u_1 = self.u_1, self.u # move over solution array index


    def run_simulation(self):
        self.u_complete = np.zeros((self.Nt, self.Nx))
        self.u_complete[0] = self.u_1
        for i in trange(1,self.Nt):
            self.advance_solution()
            self.u_complete[i] = self.u
        return self.u_complete

    def animator(self):
        x = np.linspace(0, self.L, self.Nx)
        t = np.linspace(0,self.T, self.Nt)
        fig, ax = plt.subplots()
        line, = ax.plot(x, self.u_complete[0, :], color='k', lw=2)
        v_min = np.min(self.u_complete[0])
        v_max = np.max(self.u_complete[0])
        ax.set(xlim=(x[0], x[-1]), ylim=(v_min-v_max/10, v_max))
        def animate(i):
            plt.title(f"t = {t[i]:.1f}/{t[-1]:.1f}")
            line.set_ydata(self.u_complete[i, :])
        ani = animation.FuncAnimation(fig, animate, frames = len(t)-1, interval=10)
        ani.save('503.gif')
        #plt.show()
        return ani

if __name__ == "__main__":
    I = lambda x: np.sin(np.pi * x)
    L  = 1
    T = 1
    dx = 0.1
    dt = 0.005
    c = 0
    d = 0
    PDE = PDE_solver(I, L, T, dx, dt, c, d)
    solution = PDE.run_simulation()
    PDE.animator()
