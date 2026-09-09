import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from solver import (
    center_boundary_half,
    dirichlet_matrix,
    initial_fields_half,
    interface_fluxes_half,
    neumann_matrix,
    outer_boundary,
    relax,
    small_room_boundary,
    solve_center,
    solve_outer,
)


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--omega", type=float, required=True)
parser.add_argument("--iterations", type=int, required=True)
parser.add_argument("--wall", type=float, required=True)
parser.add_argument("--heater", type=float, required=True)
parser.add_argument("--window", type=float, required=True)
parser.add_argument("--sparse", action="store_true")
parser.add_argument("--output", required=True)
args = parser.parse_args()

n = args.n
m = n // 2
h = 1.0 / n
omega = args.omega
iterations = args.iterations
wall = args.wall
heater = args.heater
window = args.window
sparse = args.sparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

u1, u2, u3, u4 = initial_fields_half(n, wall, heater, window)

# Omega 2
if rank == 0:
    A2 = dirichlet_matrix(n, 2 * n, sparse)
    gamma1 = comm.recv(source=1, tag=10)
    gamma2 = comm.recv(source=2, tag=20)
    gamma3 = comm.recv(source=3, tag=30)
    history = []

    for _ in range(iterations):
        boundary = center_boundary_half(
            n, gamma1, gamma2, gamma3, wall, heater, window
        )
        raw_u2 = solve_center(A2, boundary)

        # Compute the three neumann conditions and send them to the outer ranks
        flux1, flux3, flux4 = interface_fluxes_half(raw_u2, h)
        comm.send(flux1, dest=1, tag=11)
        comm.send(flux3, dest=2, tag=21)
        comm.send(flux4, dest=3, tag=31)

        # Receive the new dirichlet conditions
        new_gamma1 = comm.recv(source=1, tag=12)
        new_gamma2 = comm.recv(source=2, tag=22)
        new_gamma3 = comm.recv(source=3, tag=32)

        # Update change to detect convergence
        change = max(
            np.max(np.abs(new_gamma1 - gamma1)),
            np.max(np.abs(new_gamma2 - gamma2)),
            np.max(np.abs(new_gamma3 - gamma3)),
        )
        history.append(change)
        gamma1, gamma2, gamma3 = new_gamma1, new_gamma2, new_gamma3
        u2 = relax(u2, raw_u2, omega)

    # Gather the final results and save them to a file
    u1 = comm.recv(source=1, tag=41)
    u3 = comm.recv(source=2, tag=42)
    u4 = comm.recv(source=3, tag=43)
    output = Path(__file__).parent / args.output
    np.savez(
        output,
        u1=u1,
        u2=u2,
        u3=u3,
        u4=u4,
        history=np.array(history),
        h=h,
        omega=omega,
        iterations=iterations,
        wall=wall,
        heater=heater,
        window=window,
    )
    print(f"Saved {output.name} after {iterations} iterations")

# Omega 1
elif rank == 1:
    A1 = neumann_matrix(n, "right", sparse)
    boundary = outer_boundary(n, "right", wall, heater, window)
    comm.send(u1[1:-1, -1], dest=0, tag=10)

    for _ in range(iterations):
        flux1 = comm.recv(source=0, tag=11)
        raw_u1 = solve_outer(A1, boundary, flux1, h, "right")
        u1 = relax(u1, raw_u1, omega)
        comm.send(u1[1:-1, -1], dest=0, tag=12)

    comm.send(u1, dest=0, tag=41)

# Omega 3
elif rank == 2:
    A3 = neumann_matrix(n, "left", sparse)
    boundary = outer_boundary(n, "left", wall, heater, window)
    comm.send(u3[1:-1, 0], dest=0, tag=20)

    for _ in range(iterations):
        flux3 = comm.recv(source=0, tag=21)
        raw_u3 = solve_outer(A3, boundary, flux3, h, "left")
        u3 = relax(u3, raw_u3, omega)
        comm.send(u3[1:-1, 0], dest=0, tag=22)

    comm.send(u3, dest=0, tag=42)

# Omega 4
elif rank == 3:
    A4 = neumann_matrix(m, "left", sparse)
    boundary = small_room_boundary(m, wall, heater)
    comm.send(u4[1:-1, 0], dest=0, tag=30)

    for _ in range(iterations):
        flux4 = comm.recv(source=0, tag=31)
        raw_u4 = solve_outer(A4, boundary, flux4, h, "left")
        u4 = relax(u4, raw_u4, omega)
        comm.send(u4[1:-1, 0], dest=0, tag=32)

    comm.send(u4, dest=0, tag=43)
