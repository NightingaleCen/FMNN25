"""Optimization methods for Project 3, following lecture_opt.pdf.

The slide-21 Powell-Wolfe rule replaces the assignment's Fletcher algorithm.
Its defaults are slide examples, not the missing Fletcher page-37 parameters.
Quasi-Newton runs start with H = I; unsafe update denominators are skipped and
counted in skipped_updates. Line searches reject non-descent directions.
"""

from collections.abc import Callable

import numpy as np


class OptimizationProblem:
    """Store an objective and an optional analytic gradient."""

    def __init__(self, objective_function: Callable, gradient_function: Callable | None = None):
        self.objective_function = objective_function
        self.gradient_function = gradient_function

    def objective(self, x: np.ndarray) -> float:
        return float(self.objective_function(np.asarray(x, dtype=float)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.gradient_function is not None:
            return np.asarray(self.gradient_function(x), dtype=float)
        h = 1e-6
        return np.array([
            (self.objective(x + h * e) - self.objective(x - h * e)) / (2 * h)
            for e in np.eye(len(x))
        ])

    def __call__(self, x: np.ndarray) -> float:
        return self.objective(x)


class OptimizationMethod:
    """Shared iteration, gradient stopping criterion, and iterate history."""

    def __init__(self, problem, line_search=None, tol=1e-6, max_iter=1000):
        self.problem = problem
        self.line_search = line_search
        self.tol = tol
        self.max_iter = max_iter
        self.history = []

    def search_direction(self, x, g):
        raise NotImplementedError

    def update(self, delta, gamma):
        """Update a quasi-Newton approximation; classical Newton needs no update."""

    def solve(self, x0):
        x = np.asarray(x0, dtype=float).copy()
        g = self.problem.gradient(x)
        self.history = [x.copy()]
        for _ in range(self.max_iter):
            if np.linalg.norm(g) < self.tol:
                break
            s = self.search_direction(x, g)
            alpha = 1.0 if self.line_search is None else self.line_search(self.problem, x, s)
            x_new = x + alpha * s
            g_new = self.problem.gradient(x_new)
            self.update(x_new - x, g_new - g)
            x, g = x_new, g_new
            self.history.append(x.copy())
        return x


class Newton(OptimizationMethod):
    """Newton direction with a symmetrized finite-difference Hessian."""

    def __init__(self, problem, line_search=None, tol=1e-6, max_iter=1000, h=1e-6):
        super().__init__(problem, line_search, tol, max_iter)
        self.h = h

    def hessian(self, x):
        G = np.column_stack([
            (self.problem.gradient(x + self.h * e) - self.problem.gradient(x - self.h * e))
            / (2 * self.h)
            for e in np.eye(len(x))
        ])
        return 0.5 * (G + G.T)

    def search_direction(self, x, g):
        return np.linalg.solve(self.hessian(x), -g)


def exact_line_search(problem, x, s, max_iter=100):
    """Numerically minimize along the positive ray using a positive Brent bracket."""
    from scipy.optimize import minimize_scalar

    if problem.gradient(x) @ s >= 0:
        raise ValueError("Line search requires a descent direction.")
    phi = lambda alpha: problem(x + alpha * s)
    a, b = 0.0, 1.0
    fa = phi(a)
    for _ in range(max_iter):
        fb = phi(b)
        if np.isfinite(fb) and fb < fa:
            break
        b *= 0.5
    else:
        raise RuntimeError("Could not find a decreasing positive step.")
    c = 2 * b
    for _ in range(max_iter):
        fc = phi(c)
        if np.isfinite(fc) and fc > fb:
            result = minimize_scalar(phi, bracket=(a, b, c), method="brent", tol=1e-12)
            if not result.success:
                raise RuntimeError(result.message)
            return float(result.x)
        if not np.isfinite(fc):
            c = 0.5 * (b + c)
        else:
            a, b, fb = b, c, fc
            c *= 2
    raise RuntimeError("Could not bracket a finite line minimum.")


def powell_wolfe_line_search(problem, x, s, sigma=1e-2, rho=0.9, alpha0=2.0, max_iter=100):
    """Slide 21: halve for Armijo, double to bracket, then bisect for weak Wolfe."""
    if not (0 < sigma < 0.5 and sigma < rho < 1 and alpha0 > 0):
        raise ValueError("Require 0 < sigma < 1/2, sigma < rho < 1, and alpha0 > 0.")
    f0 = problem(x)
    d0 = float(problem.gradient(x) @ s)
    if d0 >= 0:
        raise ValueError("Powell-Wolfe requires a descent direction.")

    def armijo(alpha):
        value = problem(x + alpha * s)
        return np.isfinite(value) and value <= f0 + sigma * alpha * d0

    lower = float(alpha0)
    for _ in range(max_iter):
        if armijo(lower):
            break
        lower *= 0.5
    else:
        raise RuntimeError("Armijo halving did not find an acceptable step.")

    upper = lower
    for _ in range(max_iter):
        if not armijo(upper):
            break
        upper *= 2
    else:
        raise RuntimeError("Armijo doubling did not bracket an upper endpoint.")

    for _ in range(max_iter):
        if problem.gradient(x + lower * s) @ s >= rho * d0:
            return lower
        middle = 0.5 * (lower + upper)
        if middle == lower or middle == upper:
            break
        if armijo(middle):
            lower = middle
        else:
            upper = middle
    raise RuntimeError("Bisection did not satisfy the Powell-Wolfe curvature condition.")


class QuasiNewton(Newton):
    """Shared inverse-Hessian state for the five Newton subclasses."""

    def solve(self, x0):
        self.H = np.eye(len(x0))
        self.skipped_updates = 0
        return super().solve(x0)

    def search_direction(self, x, g):
        return -self.H @ g


class GoodBroyden(QuasiNewton):
    """Slide 27: inverse of the good Broyden Hessian update via Sherman-Morrison."""

    def update(self, delta, gamma):
        row = delta @ self.H
        denominator = row @ gamma
        if abs(denominator) <= 1e-12 * np.linalg.norm(row) * np.linalg.norm(gamma):
            self.skipped_updates += 1
            return
        self.H += np.outer(delta - self.H @ gamma, row) / denominator


class BadBroyden(QuasiNewton):
    """Slide 28: direct rank-one update of the inverse Hessian."""

    def update(self, delta, gamma):
        denominator = gamma @ gamma
        if denominator == 0:
            self.skipped_updates += 1
            return
        self.H += np.outer(delta - self.H @ gamma, gamma) / denominator


class SymmetricBroyden(QuasiNewton):
    """Slide 30: symmetric rank-one (SR1) inverse-Hessian update."""

    def update(self, delta, gamma):
        u = delta - self.H @ gamma
        denominator = u @ gamma
        if abs(denominator) <= 1e-8 * np.linalg.norm(u) * np.linalg.norm(gamma):
            self.skipped_updates += 1
            return
        self.H += np.outer(u, u) / denominator


class DFP(QuasiNewton):
    """Slide 31: DFP inverse-Hessian rank-two update."""

    def update(self, delta, gamma):
        Hg = self.H @ gamma
        curvature = delta @ gamma
        denominator = gamma @ Hg
        if (curvature <= 1e-12 * np.linalg.norm(delta) * np.linalg.norm(gamma)
                or denominator <= 1e-12 * np.linalg.norm(gamma) * np.linalg.norm(Hg)):
            self.skipped_updates += 1
            return
        self.H += np.outer(delta, delta) / curvature - np.outer(Hg, Hg) / denominator


class BFGS(QuasiNewton):
    """Slide 32: expanded BFGS inverse-Hessian rank-two update."""

    def update(self, delta, gamma):
        Hg = self.H @ gamma
        curvature = delta @ gamma
        if curvature <= 1e-12 * np.linalg.norm(delta) * np.linalg.norm(gamma):
            self.skipped_updates += 1
            return
        self.H += (
            (1 + (gamma @ Hg) / curvature) * np.outer(delta, delta) / curvature
            - (np.outer(delta, Hg) + np.outer(Hg, delta)) / curvature
        )


def objective(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2


def gradient(x):
    return np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])


def rosenbrock(x):
    x1, x2 = x
    return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2


def rosenbrock_gradient(x):
    x1, x2 = x
    return np.array([-400 * x1 * (x2 - x1**2) + 2 * (x1 - 1), 200 * (x2 - x1**2)])
