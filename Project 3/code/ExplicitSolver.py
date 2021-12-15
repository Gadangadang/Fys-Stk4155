import numpy as np
import matplotlib.pyplot as plt
from plot_set import *
from tqdm import trange
import matplotlib.animation as animation
import seaborn as sns


class ExplicitSolver:
    """
    Explicit solver for PDE's on the form
            d_t u(x, t) = Ad_xx u(x, t)
    """
    def __init__(self, I, L, T, dx, dt, c, d, stability=True):
        """
        Initialize the solver object. After constants are set,
        the stability criteria is checked as long as stability=True.
        The solution arrays are initialized and the initial condition
        is implemented

        Args:
            I                   (func): Initial condition function
            L                  (float): Length of the system
            T                  (float): End time
            dx                 (float): step size in spacial direction
            dt                 (float): step size in spacial direction
            c                  (float): boundary point for x = 0
            d                  (float): boundary condition for x = 1
            stability (bool, optional): Choice to check if chosen dt upholds the Neuman
                                        criteria, and corrects if it does not uphold. Defaults to True.
        """
        self.I = I
        self.L = L
        self.T = T
        self.n = 0  # set current timestep = 0
        self.c = c  # boundary points x = 0
        self.d = d  # boundary points x = Lx
        self.stability = stability
        self.alpha = 1
        # Mesh points & dx, dy, dt

        self.dt = dt
        self.dx = dx
        self.C = self.alpha*self.dt / self.dx ** 2 #stability constant, must be <= 0.5

        if self.C > 0.5 and self.stability:
            self.dt = 0.5 * self.dx ** 2
            self.C = self.dt / self.dx ** 2
            print("dt not satisfying Neuman stability criteria")
            print(f"dt is now {self.dt}")

        self.Nt = int(self.T / self.dt) + 1
        self.Nx = int(self.L / self.dx) + 1

        self.x = np.linspace(0, self.L, self.Nx)
        self.t = np.linspace(0, self.T, self.Nt)

        # Solution array (with space for ghost points)
        self.u = np.zeros((self.Nx))  # solution at t
        self.u_1 = np.zeros((self.Nx))  # solution at t - dt

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
        self.u[1:-1] = (
            self.C * (self.u_1[2:] - 2 * self.u_1[1:-1] + self.u_1[:-2])
            + self.u_1[1:-1]
        )
        self.boundary_points()
        self.n += 1  # update timestep
        self.u, self.u_1 = self.u_1, self.u  # move over solution array index

    def run_simulation(self):
        """
        Calculate solution for all t <= T

        Returns:
            array : solution
        """
        self.u_complete = np.zeros((self.Nt, self.Nx))
        self.u_complete[0] = self.u_1
        for i in range(1, self.Nt):
            self.advance_solution()
            self.u_complete[i] = self.u
        return self.u_complete

    def exact_solution(self, i):
        return np.sin(np.pi * self.x) * np.exp(-np.pi ** 2 * self.t[i])

    def plot_comparison(self, solver, name=None, title_extension=""):
        """Plots the calculated solution against the exact solution

        Args:
            solver (string): name of the type of solver
            name (string, optional): name to use to save figure. If none
                                     figure is not saved. Defaults to None.
            title_extension (str, optional): Extension for title. Defaults to "".
        """

        t_index = np.around(np.linspace(0, self.Nt - 1, 6)).astype(int)
        fig, axes = plt.subplots(2, 3, sharex="col", sharey="row")
        # fig.suptitle(f"{solver} vs Exact {title_extension}", fontsize=18)
        counter = 0
        for i in range(2):
            for j in range(3):
                axes[i, j].set_title(f"t = {self.t[t_index[counter]]:.1f}", fontsize=15)
                axes[i, j].plot(
                    self.x, self.u_complete[t_index[counter]], lw=2, label=solver
                )
                axes[i, j].plot(
                    self.x,
                    self.exact_solution(t_index[counter]),
                    "--",
                    lw=2,
                    label="Exact",
                )
                MSE = np.mean((self.exact_solution(t_index[counter])- self.u_complete[t_index[counter]])**2)
                MSE = np.where(MSE < 1e-14, 0, MSE)
                axes[i,j].text(0.15,0.25, f"MSE = {MSE:0.1e} ", fontsize = "large")
                axes[i, j].set_ylim([-0.1, 1.1])
                axes[i, j].set_xticks([0, 0.5, 1])
                counter += 1

        plt.legend(fontsize = 11)
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        ax = plt.gca()
        ax.grid(False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("$x$", fontsize = 14)
        plt.ylabel("$u$", fontsize = 14)

        # plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2, rect = (-0.05,-0.05,1,1))


        plt.subplots_adjust(hspace=0.22, wspace= 0.070)

        if name is not None:
            plt.savefig(
                f"../article/figures/{name}_dx{self.dx}.pdf", bbox_inches="tight"
            )
        plt.show()

    def animator(self, solver, name=None):
        """Animates solution

        Args:
            solver (string): name of the solver used to calculate solution
            name (string, optional): name for the animation,
                                     if no name the animation is not saved. Defaults to None.

        Returns:
            animation: the full animation
        """
        fig, ax = plt.subplots()
        (line1,) = ax.plot(self.x, self.u_complete[0][:], label=solver, lw=2)
        (line2,) = ax.plot(self.x, self.exact_solution(0), "--", label="Exact", lw=2)
        text = plt.text(
            0.6,
            0.75,
            f"MSE: {np.abs(np.mean(self.u_complete[0][:]-self.exact_solution(0))):2.2e}",
            fontsize=15,
        )
        v_min = np.min(self.u_complete[0])
        v_max = np.max(self.u_complete[0])
        ax.set(xlim=(self.x[0], self.x[-1]), ylim=(v_min - v_max / 10, v_max))

        def animate(i):
            plt.title(f"t = {self.t[i]:.1f}/{self.t[-1]:.1f}")
            text.set_text(
                f"MSE: {np.mean((self.u_complete[i][:]-self.exact_solution(i))**2):2.2e}"
            )
            line1.set_ydata(self.u_complete[i][:])
            line2.set_ydata(self.exact_solution(i))

        ani = animation.FuncAnimation(fig, animate, frames=len(self.t) - 1, interval=10)
        plt.legend()
        if name is not None:
            ani.save(f"../article/animations/PDEGif_{solver}_{name}.gif")
        plt.show()
        return ani

    def calc_err(self):
        """
        Calculates the error of the solution over time, by taking the
        difference over space and takes the mean of that error.

        Returns:
            array: contains the error per time step
        """

        error = np.zeros(len(self.t))

        for index, time in enumerate(self.t):
            # print(len(self.t), len(self.target_data))
            rel_err_ = np.abs(
                (self.target_data[index, 1:-1] - self.exact_solution(index)[1:-1])
                # / self.exact_solution(index)[1:-1]
            )
            error[index] = np.mean(rel_err_)

        return error

    def rel_err_plot(self, solver, time=None, other_data=None, other_name="", save = None):
        """
        Plots the absolute error for the solution, and possibly another data set aswell

        Args:
            solver (string): name of the first solver
            time (array, optional): time for other data set. Defaults to None.
            other_data (array, optional): data set from other solver. Defaults to None.
            other_name (str, optional): name of other data solver. Defaults to "".
        """
        self.target_data = self.u_complete
        error = self.calc_err()
        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        plt.plot(self.t, error, label=f"{solver} error")

        if isinstance(other_data, np.ndarray):
            self.target_data = other_data
            self.t = time
            error_2 = self.calc_err()
            plt.plot(time, error_2, label=f"{other_name} error")

        # plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("$t$", fontsize=14)
        plt.ylabel("MSE", fontsize=14)
        # plt.title(f"Mean absolute Error {solver} vs Exact as function of time")
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        plt.legend(fontsize = 13)
        if save is not None:
            plt.savefig(f"../article/figures/{save}_dx{self.dx}.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    I = lambda x: np.sin(np.pi * x)
    L = 1
    T = 1  # 0.5
    dx = 0.1
    dt = 0.5 * dx ** 2
    c = 0
    d = 0
    ES = ExplicitSolver(I, L, T, dx, dt, c, d)
    solution = ES.run_simulation()
    # ES.animator("Explicit solver", "001")
    # ES.rel_err_plot("Explicit solver", T)
    ES.plot_comparison("Explicit solver", name = "ExplicitPDE", title_extension=f": dx = {dx}")
