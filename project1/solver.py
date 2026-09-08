import numpy as np
from scipy.linalg import solve


def interior_index(i, j, nx):
    """Map an interior grid node to its row-major vector index."""
    return (j - 1) * (nx - 1) + i - 1


def dirichlet_matrix(nx, ny):
    """Build the inner-room matrix with Dirichlet interface conditions."""
    size = (nx - 1) * (ny - 1)
    A = np.zeros((size, size))

    for j in range(1, ny):
        for i in range(1, nx):
            row = interior_index(i, j, nx)
            A[row, row] = 4.0

            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 1 <= ni < nx and 1 <= nj < ny:
                    A[row, interior_index(ni, nj, nx)] = -1.0

    return A


def outer_unknowns(n, side):
    """List an outer room's unknown grid nodes in row-major order."""
    x_nodes = range(1, n + 1) if side == "right" else range(n)
    return [(i, j) for j in range(1, n) for i in x_nodes]


def neumann_matrix(n, side):
    """Build an outer-room matrix with a Neumann interface condition."""
    nodes = outer_unknowns(n, side)
    index = {node: k for k, node in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))

    # Interface column index
    interface_i = n if side == "right" else 0

    for node, row in index.items():
        i, j = node

        if i == interface_i:
            inward_i = n - 1 if side == "right" else 1
            A[row, row] = 3.0
            A[row, index[(inward_i, j)]] = -1.0
            for neighbor in ((i, j - 1), (i, j + 1)):
                if neighbor in index:
                    A[row, index[neighbor]] = -1.0
        else:
            A[row, row] = 4.0
            for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if neighbor in index:
                    A[row, index[neighbor]] = -1.0

    return A


def center_boundary(n, gamma1, gamma2, wall, heater, window):
    """Create the middle-room grid with its Dirichlet boundary values."""
    u = np.full((2 * n + 1, n + 1), wall)
    u[0, :] = window
    u[-1, :] = heater
    u[1:n, 0] = gamma1
    u[n + 1 : -1, -1] = gamma2

    u[0, 0] = u[0, -1] = 0.5 * (wall + window)
    u[-1, 0] = u[-1, -1] = 0.5 * (wall + heater)
    return u


def outer_boundary(n, side, wall, heater, window):
    """Create an outer-room grid with its fixed exterior boundary values."""
    u = np.full((n + 1, n + 1), wall)

    if side == "right":
        u[:, 0] = heater
        u[0, 0] = u[-1, 0] = 0.5 * (wall + heater)
        u[0, -1] = 0.5 * (wall + window)
    else:
        u[:, -1] = heater
        u[0, -1] = u[-1, -1] = 0.5 * (wall + heater)
        u[-1, 0] = 0.5 * (wall + heater)

    return u


def dirichlet_b(boundary):
    """Build the right-hand side `b` of the linear system for a Dirichlet problem."""
    ny, nx = np.array(boundary.shape) - 1
    b = np.zeros((nx - 1) * (ny - 1))

    for j in range(1, ny):
        for i in range(1, nx):
            row = interior_index(i, j, nx)
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if ni in (0, nx) or nj in (0, ny):
                    b[row] += boundary[nj, ni]

    return b


def outer_b(boundary, flux, h, side):
    """Build the right-hand side `b` for one of the two Neumann problems."""
    n = boundary.shape[0] - 1
    nodes = outer_unknowns(n, side)
    index = {node: k for k, node in enumerate(nodes)}
    interface_i = n if side == "right" else 0
    b = np.zeros(len(nodes))

    for node, row in index.items():
        i, j = node

        if i == interface_i:
            b[row] = h * flux[j - 1]
            for neighbor in ((i, j - 1), (i, j + 1)):
                if neighbor not in index:
                    ni, nj = neighbor
                    b[row] += boundary[nj, ni]
        else:
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if (ni, nj) not in index:
                    b[row] += boundary[nj, ni]

    return b


def solve_center(A, n, gamma1, gamma2, wall, heater, window):
    """Solve the middle-room Dirichlet problem and return its grid."""
    u = center_boundary(n, gamma1, gamma2, wall, heater, window)
    values = solve(A, dirichlet_b(u))

    for j in range(1, 2 * n):
        for i in range(1, n):
            u[j, i] = values[interior_index(i, j, n)]

    return u


def interface_fluxes(u2, h):
    """Compute the fluxes from the middle room into the outer rooms."""
    n = u2.shape[1] - 1
    flux1 = (u2[1:n, 1] - u2[1:n, 0]) / h
    flux3 = (u2[n + 1 : -1, -2] - u2[n + 1 : -1, -1]) / h
    return flux1, flux3


def solve_outer(A, n, flux, side, wall, heater, window):
    """Solve one outer-room Neumann problem and return its grid."""
    u = outer_boundary(n, side, wall, heater, window)
    values = solve(A, outer_b(u, flux, 1.0 / n, side))

    for value, (i, j) in zip(values, outer_unknowns(n, side)):
        u[j, i] = value

    return u


def initial_fields(n, wall, heater, window):
    """Initialize all three temperature grids and their boundaries."""
    gamma1 = np.full(n - 1, wall)
    gamma2 = np.full(n - 1, wall)
    u1 = outer_boundary(n, "right", wall, heater, window)
    u2 = center_boundary(n, gamma1, gamma2, wall, heater, window)
    u3 = outer_boundary(n, "left", wall, heater, window)
    return u1, u2, u3


def relax(old, new, omega):
    """Relax for the Dirichlet-Neumann Iteration algorithm."""
    return omega * new + (1.0 - omega) * old


def assemble_apartment(u1, u2, u3):
    """Merge the three room grids into one apartment array."""
    n = u1.shape[0] - 1
    total = np.zeros((2 * n + 1, 3 * n + 1))
    count = np.zeros_like(total)

    placements = (
        (u1, slice(0, n + 1), slice(0, n + 1)),
        (u2, slice(0, 2 * n + 1), slice(n, 2 * n + 1)),
        (u3, slice(n, 2 * n + 1), slice(2 * n, 3 * n + 1)),
    )

    for values, rows, columns in placements:
        total[rows, columns] += values
        count[rows, columns] += 1.0

    apartment = np.full_like(total, np.nan)
    apartment[count > 0] = total[count > 0] / count[count > 0]
    return apartment
