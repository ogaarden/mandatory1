import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation on the unit square"""

    L = 1.0

    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        self.N = N
        self.h = self.L / N
        xi = np.linspace(0, self.L, N + 1)
        self.xij, self.yij = np.meshgrid(xi, xi, indexing="ij", sparse=sparse)
        return self.xij, self.yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix (not scaled by h**2)
        """
        D = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], (N + 1, N + 1), "lil")
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def exact(self, t0: float) -> np.ndarray:
        """Return the exact solution evaluated on the mesh at time t0"""
        ue = sp.lambdify((x, y, t), self.ue(self.mx, self.my), "numpy")
        return np.broadcast_to(ue(self.xij, self.yij, t0), (self.N + 1, self.N + 1))

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.mx, self.my = mx, my
        self.create_mesh(N)
        D = self.D2(N).tocsr() / self.h**2
        self.D = D
        Unm1 = np.array(self.exact(0), dtype=float)
        # Taylor step using u_t(0) = 0: U^1 = U^0 + 0.5 (c dt)^2 laplace(U^0)
        Un = Unm1 + 0.5 * (self.c * self.dt) ** 2 * (D @ Unm1 + Unm1 @ D.T)
        self.apply_bcs(Un)
        return Unm1, Un

    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        return np.sqrt(self.h**2 * np.sum((u - self.exact(t0)) ** 2))

    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[0] = 0
        u[-1] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.cfl = cfl
        self.c = c
        Unm1, Un = self.initialize(N, mx, my)
        D = self.D
        dt = self.dt
        coef = (c * dt) ** 2

        data = {}
        errors = []
        if store_data > 0:
            data[0] = Unm1.copy()
            if store_data == 1:
                data[1] = Un.copy()
        else:
            errors.append(self.l2_error(Un, dt))

        for n in range(1, Nt):
            Unp1 = 2 * Un - Unm1 + coef * (D @ Un + Un @ D.T)
            self.apply_bcs(Unp1)
            Unm1, Un = Un, Unp1
            if store_data > 0:
                if (n + 1) % store_data == 0:
                    data[n + 1] = Un.copy()
            else:
                errors.append(self.l2_error(Un, (n + 1) * dt))

        if store_data > 0:
            return data
        return self.h, np.array(errors)

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        """Second order differentiation matrix with homogeneous Neumann
        conditions built in through ghost points (u_{-1} = u_1)."""
        D = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], (N + 1, N + 1), "lil")
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D

    def ue(self, mx: int, my: int) -> sp.Expr:
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self, u: np.ndarray):
        # Neumann conditions are already built into D2
        pass


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    # With mx = my and cfl = 1/sqrt(2) the scheme reproduces the exact solution
    cfl = 1 / np.sqrt(2)
    for solver in (Wave2D(), Wave2D_Neumann()):
        for m in (2, 3):
            _, err = solver(32, 20, cfl=cfl, mx=m, my=m, store_data=-1)
            assert err[-1] < 1e-12, (type(solver).__name__, m, err[-1])


if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    print("All tests passed!")