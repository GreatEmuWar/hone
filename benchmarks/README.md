# Benchmarks

This folder holds the results of `python -m hone.benchmarks`:

| File | Contents |
| --- | --- |
| [results.md](results.md) | The full suite's results table, from `make benchmark` |
| `full-summaries.json` | The same results as data |
| `quick-baseline.json` | The quick suite's results, which CI compares every change against |

## What the suite measures

Each optimizer runs on each problem with 10 seeds and the same budget: 1,000 evaluations per dimension. The table reports the median **error**, the best value found minus the problem's known optimum, and the share of runs **solved** to an error of 0.0001 or less.

Every variable lies in [-5, 5]. Each problem is shifted so its optimum sits at a fixed point away from the centre, so an optimizer that drifts toward the centre gains nothing.

The plain functions are separable: each variable affects the value on its own, so an optimizer can improve one variable at a time. Engineering problems rarely behave like that, so ellipsoid, Rosenbrock, Rastrigin and Ackley also appear rotated, which mixes the variables together. Sphere is the same under any rotation and appears only once.

| Problem | What it tests |
| --- | --- |
| sphere | Plain convergence on a smooth bowl |
| ellipsoid | Variables on very different scales: axis weights range from 1 to 1,000,000 |
| rosenbrock | Following a long, curved valley |
| rastrigin | Escaping a regular grid of local minima |
| ackley | Finding a narrow hole in an almost flat surface |
| rotated-* | The same functions turned by a fixed random rotation, so the variables interact rather than each acting on its own |
| integer-sphere | Integer variables only |
| mixed-rastrigin | Half continuous and half integer variables |
| material-selection | A categorical choice that moves where the best continuous design is. Each material adds a base cost; titanium, at cost 0, gives the optimum |

## What the 0.5.0 results show

CMA-ES runs on the 33 problems without choice parameters. Against the genetic optimizer it wins 26, ties 3 (the integer spheres, which both solve) and loses 4: plain Rastrigin in 5 and 10 dimensions and plain mixed Rastrigin in 5 and 10 dimensions.

- **Those four wins come from separability.** Uniform crossover takes each variable from one parent or the other, which works well when variables act independently. Rotated, the genetic optimizer's median error on Rastrigin rises from 0.228 to 22.4 in 10 dimensions, about a hundredfold, while CMA-ES's barely changes (5.47 to 6.47). CMA-ES wins all 12 rotated problems.
- **Smooth and badly scaled problems: CMA-ES by far, rotated or not.** It solves sphere, ellipsoid, Rosenbrock and Ackley to between about 1e-11 and 1e-17 in every dimension. On rotated-ellipsoid-10d the genetic optimizer's median error is 7,050.
- **Rastrigin stays hard for CMA-ES at this budget.** Its deep, dense local minima trap it; solving it reliably takes much larger populations than 1,000 evaluations per dimension allows the restarts to reach.
- **Categorical problems: only the genetic optimizer.** On material-selection-10d its median error is exactly 1, the base cost of aluminium: most runs settle on the wrong material.
- **The genetic optimizer against random search:** it loses ellipsoid-2d, rotated-ellipsoid-2d, rotated-rosenbrock-2d and rotated-rastrigin-2d, where its fixed mutation step is too coarse for the precision those problems need.

Since 0.5.0 every optimizer gets exactly the same number of evaluations. Before, the genetic optimizer used about 10% fewer than its budget, because it does not evaluate its elites again.
