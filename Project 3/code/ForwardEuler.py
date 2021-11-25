import numpy as np
import matplotlib.pyplot as plt


class wave_solver:
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
        self.Nx = int(Lx/dx)
        self.C = dt/dx**2

        self.x = np.linspace(0, self.L, self.Nx)

        # Solution array (with space for ghost points)
        self.u = np.zeros((self.Nx))   # solution at t
        self.u_1 = np.zeros((self.Nx)) # solution at t - dt

        # Initial conditions
        self.u[1:-1] = I(self.x[1:-1])

        # Initialize ghost points
        self.boundary_points(self.u)


    def boundary_points(self, u):
        """
        Initialize boundary points.
        """
        u[0] = 0
        u[-1] = 0

    def __call__(self):
        """
        Returns last solution without ghost points.
        """
        return self.u[1:-1,1:-1]

    def advance_solution(self):
        """
        Advance solution to get u = u[n+1].
        """
        I = self.I; x = self.x; dt = self.dt; C = self.C

        if self.n == 0: # Modified scheme for n=0
            self.u_1 = self.u.copy()  # move over solution array index
            self.u[1:-1] =  C*(I(x)-2*self.u_1[1:-1] + self.u[:-2]) + self.u[1:-1]
            self.boundary_points(self.u)
            self.n += 1 # update timestep

        else: # General scheme
            self.u, self.u_1 = self.u_1, self.u # move over solution array index
            self.u[1:-1] = C*(self.u_1[2:-1]-2*self.u_1[1:-1] + self.u[:-2]) + self.u[1:-1]
            self.boundary_points(self.u)
            self.n += 1 # update timestep


    def run_simulation(self):
        self.u_complete = np.zeros((self.Nt, self.Nx))
        for i in trange(self.Nt):
            self.u_complete[i] = self.u[1:-1]
            self.advance_solution()
        return self.u_complete
if __name__ == "__main__":
    
