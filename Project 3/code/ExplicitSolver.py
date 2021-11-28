import numpy as np
import matplotlib.pyplot as plt
from plot_set import *
from tqdm import trange
import matplotlib.animation as animation
import seaborn as sns


class ExplicitSolver:
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
        self.t = np.linspace(0,self.T, self.Nt)

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

    def exact_solution(self,i):
        return np.sin(np.pi*self.x)*np.exp(-np.pi**2*self.t[i])

    def plot_comparison(self, solver, name = None, title_extension = ""):

        t_index = np.around(np.linspace(0,self.Nt-1, 6)).astype(int)
        fig, axes = plt.subplots(2, 3, sharex='col', sharey='row')
        fig.suptitle(f"{solver} vs Exact {title_extension}", fontsize = 18)
        counter = 0
        for i in range(2):
            for j in range(3):
                axes[i,j].set_title(f"t = {self.t[t_index[counter]]:.1f}", fontsize = 15)
                axes[i,j].plot(self.x, self.u_complete[t_index[counter]], lw=2, label = solver)
                axes[i,j].plot(self.x, self.exact_solution(t_index[counter]), "--", lw=2, label = "Exact")
                axes[i,j].set_ylim([-0.1,1.1])
                counter += 1
        plt.subplots_adjust(hspace = 2, wspace= 0.11)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        plt.legend()
        if name is not None:
            plt.savefig(f"../article/figures/ExplicitPDE_dx{self.dx}.pdf", bbox_inches="tight")
        plt.show()

    def animator(self,solver, name = None):
        fig, ax = plt.subplots()
        line1, = ax.plot(self.x, self.u_complete[0][:], label = solver,  lw=2)
        line2, = ax.plot(self.x, self.exact_solution(0), "--", label = "Exact",  lw=2)
        text = plt.text(0.6, 0.75, f"MSE: {np.abs(np.mean(self.u_complete[0][:]-self.exact_solution(0))):2.2e}", fontsize = 15)
        v_min = np.min(self.u_complete[0])
        v_max = np.max(self.u_complete[0])
        ax.set(xlim=(self.x[0], self.x[-1]), ylim=(v_min-v_max/10, v_max))
        def animate(i):
            plt.title(f"t = {self.t[i]:.1f}/{self.t[-1]:.1f}")
            text.set_text(f"MSE: {np.abs(np.mean(self.u_complete[i][:]-self.exact_solution(i))):2.2e}")
            line1.set_ydata(self.u_complete[i][:])
            line2.set_ydata(self.exact_solution(i))
        ani = animation.FuncAnimation(fig, animate, frames = len(self.t)-1, interval=10)
        plt.legend()
        if name is not None:
            ani.save("../article/animations/ExplicitPDEGif.gif")
        plt.show()
        return ani

if __name__ == "__main__":
    I = lambda x: np.sin(np.pi * x)
    L  = 1
    T = 0.5
    dx = 1/100
    dt = 0.5*dx**2
    c = 0
    d = 0
    ES = ExplicitSolver(I, L, T, dx, dt, c, d)
    solution = ES.run_simulation()
    #PDE.animator()
    ES.plot_comparison("Explicit solver", title_extension = f": dx = {dx}")
