import numpy as np
from scipy.linalg import solve
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.linalg import spsolve


def interior_index(i, j, nx):
    """Map an interior grid node to its row-major vector index."""
    return (j - 1) * (nx - 1) + i - 1


def build_matrix(rows, columns, values, size, sparse):
    """Assemble collected stencil entries as a dense or sparse matrix."""
    if sparse:
        return coo_matrix((values, (rows, columns)), shape=(size, size)).tocsr()

    A = np.zeros((size, size))
    A[rows, columns] = values
    return A


def solve_system(A, b):
    """Solve `A x = b` with the solver matching the matrix format."""
    return spsolve(A, b) if issparse(A) else solve(A, b)


def dirichlet_matrix(nx, ny, sparse=False):
    """Build the inner-room matrix with Dirichlet interface conditions."""
    rows, columns, values = [], [], []

    for j in range(1, ny):
        for i in range(1, nx):
            row = interior_index(i, j, nx)
            rows.append(row)
            columns.append(row)
            values.append(4.0)

            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 1 <= ni < nx and 1 <= nj < ny:
                    rows.append(row)
                    columns.append(interior_index(ni, nj, nx))
                    values.append(-1.0)

    return build_matrix(rows, columns, values, (nx - 1) * (ny - 1), sparse)


def outer_unknowns(n, side):
    """List an outer room's unknown grid nodes in row-major order."""
    x_nodes = range(1, n + 1) if side == "right" else range(n)
    return [(i, j) for j in range(1, n) for i in x_nodes]


def neumann_matrix(n, side, sparse=False):
    """Build an outer-room matrix with a Neumann interface condition."""
    nodes = outer_unknowns(n, side)
    index = {node: k for k, node in enumerate(nodes)}
    rows, columns, values = [], [], []

    # Interface column index
    interface_i = n if side == "right" else 0

    for node, row in index.items():
        i, j = node

        if i == interface_i:
            inward_i = n - 1 if side == "right" else 1
            neighbors = [(inward_i, j), (i, j - 1), (i, j + 1)]
            diagonal = 3.0
        else:
            neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            diagonal = 4.0

        rows.append(row)
        columns.append(row)
        values.append(diagonal)

        for neighbor in neighbors:
            if neighbor in index:
                rows.append(row)
                columns.append(index[neighbor])
                values.append(-1.0)

    return build_matrix(rows, columns, values, len(nodes), sparse)


def center_boundary(n, gamma1, gamma2, wall, heater, window):
    """
    Create the middle-room grid with its Dirichlet boundary values.

    corner nodes are set to the average of the two adjacent boundaries.
    """
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


def center_boundary_half(n, gamma1, gamma2, gamma3, wall, heater, window):
    """Create the middle-room grid for the 2.5-room layout.

    The right edge now carries two interfaces: Gamma2 against Omega3 on
    the upper half, and Gamma3 against Omega4 between y = 1/2 and y = 1.
    The remaining lower part of that edge is a normal wall.
    """
    m = n // 2
    u = np.full((2 * n + 1, n + 1), wall)
    u[0, :] = window
    u[-1, :] = heater
    u[1:n, 0] = gamma1
    u[n + 1 : -1, -1] = gamma2
    u[m + 1 : n, -1] = gamma3

    # Omega4's heater floor meets the wall below it at this single node
    u[m, -1] = 0.5 * (wall + heater)
    u[0, 0] = u[0, -1] = 0.5 * (wall + window)
    u[-1, 0] = u[-1, -1] = 0.5 * (wall + heater)
    return u


def small_room_boundary(m, wall, heater):
    """Create Omega4's grid: a heater along the floor, normal walls elsewhere."""
    u = np.full((m + 1, m + 1), wall)
    u[0, :] = heater
    u[0, 0] = u[0, -1] = 0.5 * (wall + heater)
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
    """Build the right-hand side `b` for one of the Neumann problems."""
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


def solve_center(A, boundary):
    """Solve a middle-room Dirichlet problem on a prepared boundary grid."""
    ny, nx = np.array(boundary.shape) - 1
    u = boundary.copy()
    values = solve_system(A, dirichlet_b(boundary))

    for j in range(1, ny):
        for i in range(1, nx):
            u[j, i] = values[interior_index(i, j, nx)]

    return u


def solve_outer(A, boundary, flux, h, side):
    """Solve one outer-room Neumann problem on a prepared boundary grid."""
    n = boundary.shape[0] - 1
    u = boundary.copy()
    values = solve_system(A, outer_b(boundary, flux, h, side))

    for value, (i, j) in zip(values, outer_unknowns(n, side)):
        u[j, i] = value

    return u


def interface_fluxes(u2, h):
    """Compute the fluxes from the middle room into the two outer rooms."""
    n = u2.shape[1] - 1
    flux1 = (u2[1:n, 1] - u2[1:n, 0]) / h
    flux3 = (u2[n + 1 : -1, -2] - u2[n + 1 : -1, -1]) / h
    return flux1, flux3


def interface_fluxes_half(u2, h):
    """Compute the fluxes from the middle room for the 2.5-room layout."""
    n = u2.shape[1] - 1
    m = n // 2
    flux1 = (u2[1:n, 1] - u2[1:n, 0]) / h
    flux3 = (u2[n + 1 : -1, -2] - u2[n + 1 : -1, -1]) / h
    flux4 = (u2[m + 1 : n, -2] - u2[m + 1 : n, -1]) / h
    return flux1, flux3, flux4


def initial_fields(n, wall, heater, window):
    """Initialize the three temperature grids of the 2-room apartment."""
    gamma1 = np.full(n - 1, wall)
    gamma2 = np.full(n - 1, wall)
    u1 = outer_boundary(n, "right", wall, heater, window)
    u2 = center_boundary(n, gamma1, gamma2, wall, heater, window)
    u3 = outer_boundary(n, "left", wall, heater, window)
    return u1, u2, u3


def initial_fields_half(n, wall, heater, window):
    """Initialize the four temperature grids of the 2.5-room apartment."""
    m = n // 2
    gamma1 = np.full(n - 1, wall)
    gamma2 = np.full(n - 1, wall)
    gamma3 = np.full(m - 1, wall)
    u1 = outer_boundary(n, "right", wall, heater, window)
    u2 = center_boundary_half(n, gamma1, gamma2, gamma3, wall, heater, window)
    u3 = outer_boundary(n, "left", wall, heater, window)
    u4 = small_room_boundary(m, wall, heater)
    return u1, u2, u3, u4


def relax(old, new, omega):
    """Relax for the Dirichlet-Neumann Iteration algorithm."""
    return omega * new + (1.0 - omega) * old


def assemble(placements, shape):
    """Merge room grids placed at given (row, column) offsets into one array."""
    total = np.zeros(shape)
    count = np.zeros(shape)

    for values, row0, column0 in placements:
        rows = slice(row0, row0 + values.shape[0])
        columns = slice(column0, column0 + values.shape[1])
        total[rows, columns] += values
        count[rows, columns] += 1.0

    apartment = np.full(shape, np.nan)
    apartment[count > 0] = total[count > 0] / count[count > 0]
    return apartment


def assemble_apartment(u1, u2, u3):
    """Merge the three room grids of the 2-room apartment."""
    n = u1.shape[0] - 1
    placements = ((u1, 0, 0), (u2, 0, n), (u3, n, 2 * n))
    return assemble(placements, (2 * n + 1, 3 * n + 1))


def assemble_apartment_half(u1, u2, u3, u4):
    """Merge the four room grids of the 2.5-room apartment."""
    n = u1.shape[0] - 1
    m = n // 2
    placements = ((u1, 0, 0), (u2, 0, n), (u3, n, 2 * n), (u4, m, 2 * n))
    return assemble(placements, (2 * n + 1, 3 * n + 1))
