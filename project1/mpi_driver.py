import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from solver import (
    dirichlet_matrix,
    initial_fields,
    interface_fluxes,
    neumann_matrix,
    relax,
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
parser.add_argument("--output", required=True)
args = parser.parse_args()

n = args.n
h = 1.0 / n
omega = args.omega
iterations = args.iterations
wall = args.wall
heater = args.heater
window = args.window

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

u1, u2, u3 = initial_fields(n, wall, heater, window)

# Omega 2
if rank == 0:
    A2 = dirichlet_matrix(n, 2 * n)
    gamma1 = comm.recv(source=1, tag=10)
    gamma2 = comm.recv(source=2, tag=20)
    history = []

    for _ in range(iterations):
        raw_u2 = solve_center(A2, n, gamma1, gamma2, wall, heater, window)

        # Compute the neumann conditions directly and send them to rank1 and rank2
        flux1, flux3 = interface_fluxes(raw_u2, h)
        comm.send(flux1, dest=1, tag=11)
        comm.send(flux3, dest=2, tag=21)

        # Receive the new dirichlet condition
        new_gamma1 = comm.recv(source=1, tag=12)
        new_gamma2 = comm.recv(source=2, tag=22)

        # Update change to detect convergence
        change = max(
            np.max(np.abs(new_gamma1 - gamma1)),
            np.max(np.abs(new_gamma2 - gamma2)),
        )
        history.append(change)
        gamma1, gamma2 = new_gamma1, new_gamma2
        u2 = relax(u2, raw_u2, omega)

    # Gather the final results and save them to a file
    u1 = comm.recv(source=1, tag=31)
    u3 = comm.recv(source=2, tag=32)
    output = Path(__file__).parent / args.output
    np.savez(
        output,
        u1=u1,
        u2=u2,
        u3=u3,
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
    A1 = neumann_matrix(n, "right")
    comm.send(u1[1:-1, -1], dest=0, tag=10)

    for _ in range(iterations):
        flux1 = comm.recv(source=0, tag=11)
        raw_u1 = solve_outer(A1, n, flux1, "right", wall, heater, window)
        u1 = relax(u1, raw_u1, omega)
        comm.send(u1[1:-1, -1], dest=0, tag=12)

    comm.send(u1, dest=0, tag=31)

# Omega 3
elif rank == 2:
    A3 = neumann_matrix(n, "left")
    comm.send(u3[1:-1, 0], dest=0, tag=20)

    for _ in range(iterations):
        flux3 = comm.recv(source=0, tag=21)
        raw_u3 = solve_outer(A3, n, flux3, "left", wall, heater, window)
        u3 = relax(u3, raw_u3, omega)
        comm.send(u3[1:-1, 0], dest=0, tag=22)

    comm.send(u3, dest=0, tag=32)
