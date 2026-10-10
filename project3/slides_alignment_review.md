# Project 3: Differences between the original solutions and the lecture slides

## Review scope

- Lecture slides: `lecture_opt.pdf`, especially slides 10-11, 15, 17, 20-21, 23, and 25-34.
- Assignment instructions: [project03.pdf](project03.pdf).
- Original code: commit `a10e3ee`; the subsequent Task 6 implementation: commit `a34102c`.
- Current refactored version: [main.ipynb](main.ipynb) and [optimization.py](optimization.py), covering Tasks 1-9.

Here, a departure from the slides means a difference in the core algorithm or its constraints. Differences in variable names, class structure, plot styling, or example initial values are not automatically algorithmic errors.

## Original tasks requiring changes

After the Task 6 implementation was added, there were two independent departures from the lecture methods: **Task 4 and Task 6**. Task 5 uses the Task 4 line search and is therefore affected, but it is not counted as a third independent issue.

| Task | Original version | Current refactored version |
|---|---|---|
| 1-2 | The slides do not prescribe a class structure; no conflict with the lecture methods was found | Retains the problem class, shared iteration, and common line-search interface |
| 3 | Newton direction, finite-difference Hessian, and symmetrization follow the requirements | Retained |
| 4 | Unrestricted scalar minimization can return a negative step | Constructs a bracket with nonnegative endpoints before numerical minimization |
| 5 | The Rosenbrock experiment follows the requirements but uses the original Task 4 search | Retains the experiment with the corrected line search |
| 6 | `a10e3ee` contains only an explanation; `a34102c` adds conditions and a search procedure different from slide 21 | Uses the lecture's Powell-Wolfe algorithm |
| 7-9 | Not completed in either original commit | Experiments and all five Quasi-Newton methods are implemented |

### Task 4: Exact line search lacks the nonnegative-step constraint

Slide 15 defines exact line search over nonnegative step lengths:

$$
\alpha_k\in\operatorname*{arg\,min}_{\alpha\geq 0} f(x_k+\alpha s_k).
$$

The original implementation calls:

```python
minimize_scalar(lambda alpha: problem(x + alpha * s)).x
```

This call does not enforce `alpha >= 0`, so it does not guarantee the step constraint in the slides. A positive step in the Rosenbrock experiment does not establish that the constraint always holds.

The following counterexample was checked:

$$
f(t)=t^4+5t^3-4t^2-t,\qquad x=0,\quad s=1.
$$

Here, $f'(0)s=-1<0$, so the direction is a descent direction. The original search returns approximately `-4.210863`; the current search along the nonnegative ray returns approximately `0.565795`.

The current implementation first constructs a nonnegative bracket and then applies Brent's method. Accuracy and the selected minimum still depend on the function and bracket; this is not a guarantee of a global minimum for every nonconvex function.

### Task 6: The updated version uses different Wolfe conditions and search steps

Slides 20-21 use the Armijo condition and the weak Wolfe curvature condition:

$$
\phi(\alpha)\leq\phi(0)+\sigma\alpha\phi'(0),
\qquad
\phi'(\alpha)\geq\rho\phi'(0).
$$

The lecture algorithm proceeds as follows:

1. Halve the lower endpoint until it satisfies Armijo.
2. Double the upper endpoint until it fails Armijo.
3. Bisect the interval until the lower endpoint also satisfies weak Wolfe.

The implementation added in `a34102c` instead uses strong Wolfe:

$$
\phi(\alpha)\leq\phi(0)+\rho\alpha\phi'(0),
\qquad
|\phi'(\alpha)|\leq-\sigma\phi'(0).
$$

The names `rho` and `sigma` are exchanged relative to the slides, which is not itself an error. The substantive differences are the absolute-value curvature condition and the different bracketing/sectioning procedure. This line search is not mathematically wrong simply because it differs from the slides, but it is not a direct implementation of slide 21. The relevant Fletcher pages were unavailable, so exact agreement with Algorithms (2.6.2) and (2.6.4) has not been verified.

Two further points need clarification in the updated version:

- The Markdown states a sufficient-decrease parameter of `0.01`, but the test uses `1e-4`. The lecture example uses `sigma=0.01` and `rho=0.9`.
- Starting from `alpha0=1`, `alpha_next = 2 * alpha - alpha_prev` produces the arithmetic sequence `1, 2, 3, ...`. For $\phi(\alpha)=(\alpha-10000)^2$, the default limit of 100 iterations produces `Bracketing failed to converge`, whereas the lecture's doubling procedure finds an acceptable step.

The updated standalone Rosenbrock test passes: the step is `0.0009765625`, and both Armijo and strong Wolfe hold. Passing this example does not make the two algorithms equivalent.

## Alignment of the current completed tasks

- Task 3 solves the linear system for the Newton direction, uses a finite-difference Hessian with symmetrization, and stops using the gradient norm, following slides 10-11.
- Tasks 4-5 correct the step constraint while retaining exact line search and the Rosenbrock experiment, following slide 15.
- Tasks 6-7 implement and separately check Powell-Wolfe using slides 20-21 and their example parameters.
- Task 8 supplies the same line-search interface to Newton. It first records the non-descent direction encountered with initial step `2`, then compares convergence using initial step `1`. Slide 21 permits any positive initial step; `2` is only an example, so using `1` is not an algorithmic departure. No automatic direction replacement or Hessian modification is applied.
- Task 9 implements good Broyden, bad Broyden, SR1, DFP, and BFGS using the formulas on slides 27, 28, 30, 31, and 32. Unsafe denominators or unsuitable curvature cause an update to be skipped and counted. This is additional numerical handling, rather than a step stated in the slides.

Reusable classes and functions have been moved to `optimization.py`. The notebook retains explanations, experiments, results, and plots. The saved notebook contains 27 cells; all 13 code cells were executed in order without error outputs.

## Difference from the written assignment requirements

Assignment Task 6 specifies Fletcher's algorithm, and Task 7 specifies the parameters on page 37 of that book. Under the agreed scope, the current version substitutes the algorithm and example parameters from slide 21.

Therefore, **alignment with the lecture slides** and **strict compliance with the original Fletcher-based assignment** are separate judgments. This difference is explicitly documented; the lecture parameters must not be presented as verified Fletcher page-37 parameters.
