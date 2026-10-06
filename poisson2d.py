import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

x, y = sp.symbols("x,y")


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float):
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh"""
        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return 1D second derivative matrix (central differences)"""
        h = self.p.L / N
        D = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], (N + 1, N + 1), "lil")
        # Second order one-sided stencils on the boundary rows
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D / h**2

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator"""
        D2 = self.D2(N)
        I = sparse.eye(N + 1)
        return (sparse.kron(D2, I) + sparse.kron(I, D2)).tolil()

    def get_boundary_indices(self, N: int) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((N + 1, N + 1), dtype=bool)
        B[1:-1, 1:-1] = False
        return np.where(B.ravel())[0]

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b"""
        A = self.laplace(N)
        bnds = self.get_boundary_indices(N)
        # Replace boundary rows with identity rows (Dirichlet)
        for i in bnds:
            A.rows[i] = [i]
            A.data[i] = [1.0]
        A = A.tocsr()

        xij, yij = self.create_mesh(N)
        b = self.meshfunction(f, xij, yij)
        uij = self.meshfunction(ue, xij, yij)
        b.ravel()[bnds] = uij.ravel()[bnds]
        return A, b

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        """Return Sympy function as mesh function"""
        shape = (xij.shape[0], yij.shape[1])
        values = sp.lambdify((x, y), u, "numpy")(xij, yij)
        return np.array(np.broadcast_to(values, shape), dtype=float)

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error"""
        N = u.shape[0] - 1
        h = self.p.L / N
        xij, yij = self.create_mesh(N)
        uej = self.meshfunction(ue, xij, yij)
        return np.sqrt(h * h * np.sum((u - uej) ** 2))

    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution"""
        A, b = self.assemble(N, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

    @staticmethod
    def _lagrange_weights(nodes: np.ndarray, xp: float) -> np.ndarray:
        """Lagrange basis functions for the given nodes evaluated at xp"""
        w = np.ones(len(nodes))
        for j in range(len(nodes)):
            for k in range(len(nodes)):
                if k != j:
                    w[j] *= (xp - nodes[k]) / (nodes[j] - nodes[k])
        return w

    def eval(self, U: np.ndarray, x: float, y: float) -> float:
        """Return u(x, y) using cubic Lagrange interpolation on a 4x4 stencil"""
        N = U.shape[0] - 1
        h = self.p.L / N
        xi = self.p.create_mesh(N)

        def stencil(p):
            i = int(np.floor(p / h))
            i0 = min(max(i - 1, 0), N - 3)
            return np.arange(i0, i0 + 4)

        ix, iy = stencil(x), stencil(y)
        wx = self._lagrange_weights(xi[ix], x)
        wy = self._lagrange_weights(xi[iy], y)
        return float(wx @ U[np.ix_(ix, iy)] @ wy)


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h, y: 1 - h / 2}).n()) < 1e-3


if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")