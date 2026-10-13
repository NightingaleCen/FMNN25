"""Newton and quasi-Newton methods with exact and inexact line search."""

import numpy as np
from scipy.optimize import minimize_scalar


class OptimizationProblem:
    """Class for optimization problems with an objective and optional gradient."""

    def __init__(self, objective_function, gradient_function=None):
        self.objective_function = objective_function
        self.gradient_function = gradient_function

    def objective(self, x):
        return float(self.objective_function(np.asarray(x, dtype=float)))

    def gradient(self, x):
        x = np.asarray(x, dtype=float)
        if self.gradient_function is not None:
            return np.asarray(self.gradient_function(x), dtype=float)
        h = 1e-6
        return np.array([
            (self.objective(x + h * e) - self.objective(x - h * e)) / (2 * h) # central difference
            for e in np.eye(len(x))
        ])

    def __call__(self, x):
        return self.objective(x)


class OptimizationMethod:
    """A general optimization method class for quasi-newton methods with line search."""

    def __init__(self, problem, line_search=None, tol=1e-6, max_iter=1000):
        self.problem = problem
        self.line_search = line_search
        self.tol = tol
        self.max_iter = max_iter
        self.history = []
        self.converged = False

    def search_direction(self, x, g):
        ''' To be implemented in subclasses. Returns the search direction given the current point x and gradient g.'''
        raise NotImplementedError

    def update(self, delta, gamma):
        ''' To be implemented in subclasses. Updates the internal state of the optimization method.'''
        pass

    def solve(self, x0):
        x = np.asarray(x0, dtype=float).copy()
        g = self.problem.gradient(x)
        self.history = [x.copy()]
        self.converged = False
        for _ in range(self.max_iter):
            if np.linalg.norm(g) < self.tol:
                break
            s = self.search_direction(x, g)
            alpha = 1.0 if self.line_search is None else self.line_search(self.problem, x, s) # use line search if provided
            x_new = x + alpha * s # iterate here
            g_new = self.problem.gradient(x_new)
            self.update(x_new - x, g_new - g)
            x, g = x_new, g_new
            self.history.append(x.copy())
        self.converged = bool(np.linalg.norm(g) < self.tol)
        return x


class Newton(OptimizationMethod):
    """Newton's method with a finite-difference Hessian."""

    def __init__(self, problem, line_search=None, tol=1e-6, max_iter=1000, h=1e-6):
        super().__init__(problem, line_search, tol, max_iter)
        self.h = h

    def hessian(self, x):
        G = np.column_stack([
            (self.problem.gradient(x + self.h * e) - self.problem.gradient(x - self.h * e))
            / (2 * self.h)
            for e in np.eye(len(x))
        ])
        return 0.5 * (G + G.T) # symmetrizing step

    def search_direction(self, x, g):
        return np.linalg.solve(self.hessian(x), -g)


def exact_line_search(problem, x, s, max_iter=100):
    """Minimize f(x + alpha*s) for alpha >= 0."""
    if problem.gradient(x) @ s >= 0: # ensure descent direction
        raise ValueError("Line search requires a descent direction.")
    phi = lambda alpha: problem(x + alpha * s)
    a, b = 0.0, 1.0
    fa = phi(a)
    for _ in range(max_iter):
        fb = phi(b)
        if np.isfinite(fb) and fb < fa: # find a point that is lower than the starting point a
            break
        b *= 0.5
    else:
        raise RuntimeError("No decreasing step found.")
    
    c = 2 * b
    for _ in range(max_iter):
        fc = phi(c)
        if np.isfinite(fc) and fc >= fb: # find a point that is higher than (or equal) the previous point b
            result = minimize_scalar(
                phi, bounds=(a, c), method="bounded", options={"xatol": 1e-12},
            )
            if not result.success:
                raise RuntimeError(result.message)
            return float(result.x)
        if not np.isfinite(fc):
            c = 0.5 * (b + c)
        else:
            a, b, fb = b, c, fc
            c *= 2
    raise RuntimeError("Could not bracket a minimum.")


def fletcher_line_search(
    problem, x, s, rho=0.01, sigma=0.1, alpha0=1.0,
    tau1=9.0, tau2=0.1, tau3=0.5, f_lower=None, max_iter=100,
):
    """Line search algorithm from Fletcher's book"""
    f0 = problem(x)
    d0 = float(problem.gradient(x) @ s)
    if d0 >= 0:
        raise ValueError("Line search requires a descent direction.")
    mu = np.inf if f_lower is None else (f_lower - f0) / (rho * d0)

    # (2.6.2) bracketing phase
    previous, f_previous = 0.0, f0
    alpha = min(float(alpha0), mu)
    for _ in range(max_iter):
        value = problem(x + alpha * s)

        if f_lower is not None and value <= f_lower:
            return alpha
        
        if (not np.isfinite(value) or value > f0 + rho * alpha * d0 # condition 1
                or value >= f_previous):
            a, b, fa = previous, alpha, f_previous
            break # terminate B

        slope = float(problem.gradient(x + alpha * s) @ s)
        if abs(slope) <= -sigma * d0: # condition 2
            return alpha
        
        if slope >= 0:
            a, b, fa = alpha, previous, value
            break # terminate B

        next_alpha = min(mu, alpha + tau1 * (alpha - previous)) # we choose the upper bound from (2.6.2)
        previous, f_previous, alpha = alpha, value, next_alpha
    else:
        raise RuntimeError("Could not bracket an acceptable step.")

    # (2.6.3) sectioning phase
    for _ in range(max_iter):
        alpha = a + np.clip(0.5, tau2, 1 - tau3) * (b - a) # choose middle point if possible, otherwise choose the lower or upper bound

        value = problem(x + alpha * s)
        if f_lower is not None and value <= f_lower: # end early if we reach an acceptable value
            return alpha
        
        if value > f0 + rho * alpha * d0 or value >= fa: # condition 1
            b = alpha
            continue

        slope = float(problem.gradient(x + alpha * s) @ s)
        if abs(slope) <= -sigma * d0: # condition 2
            return alpha
        
        if slope * (b - a) >= 0:
            b = a
        a, fa = alpha, value


class QuasiNewton(Newton):
    """Use an inverse-Hessian approximation, starting from the identity."""

    def solve(self, x0):
        self.H = np.eye(len(x0))
        return super().solve(x0)

    def search_direction(self, x, g):
        return -self.H @ g


class GoodBroyden(QuasiNewton):
    """Good Broyden update, inverted with Sherman-Morrison."""

    def update(self, delta, gamma):
        row = delta @ self.H
        denominator = row @ gamma
        if abs(denominator) <= 1e-12 * np.linalg.norm(row) * np.linalg.norm(gamma):
            return
        self.H += np.outer(delta - self.H @ gamma, row) / denominator


class BadBroyden(QuasiNewton):
    """Rank-one update of the inverse Hessian."""

    def update(self, delta, gamma):
        denominator = gamma @ gamma
        if denominator == 0:
            return
        self.H += np.outer(delta - self.H @ gamma, gamma) / denominator


class SymmetricBroyden(QuasiNewton):
    """Symmetric rank-one (SR1) update."""

    def update(self, delta, gamma):
        u = delta - self.H @ gamma
        denominator = u @ gamma
        if abs(denominator) <= 1e-8 * np.linalg.norm(u) * np.linalg.norm(gamma):
            return
        self.H += np.outer(u, u) / denominator


class DFP(QuasiNewton):
    """DFP inverse-Hessian update."""

    def update(self, delta, gamma):
        Hg = self.H @ gamma
        curvature = delta @ gamma
        denominator = gamma @ Hg
        if (curvature <= 1e-12 * np.linalg.norm(delta) * np.linalg.norm(gamma)
                or denominator <= 1e-12 * np.linalg.norm(gamma) * np.linalg.norm(Hg)):
            return
        self.H += np.outer(delta, delta) / curvature - np.outer(Hg, Hg) / denominator


class BFGS(QuasiNewton):
    """BFGS inverse-Hessian update."""

    def update(self, delta, gamma):
        Hg = self.H @ gamma
        curvature = delta @ gamma
        if curvature <= 1e-12 * np.linalg.norm(delta) * np.linalg.norm(gamma):
            return
        self.H += (
            (1 + (gamma @ Hg) / curvature) * np.outer(delta, delta) / curvature
            - (np.outer(delta, Hg) + np.outer(Hg, delta)) / curvature
        )
