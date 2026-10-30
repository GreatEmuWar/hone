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

| Problem | What it tests |
| --- | --- |
| sphere | Plain convergence on a smooth bowl |
| ellipsoid | Variables on very different scales: axis weights range from 1 to 1,000,000 |
| rosenbrock | Following a long, curved valley |
| rastrigin | Escaping a regular grid of local minima |
| ackley | Finding a narrow hole in an almost flat surface |
| integer-sphere | Integer variables only |
| mixed-rastrigin | Half continuous and half integer variables |
| material-selection | A categorical choice that moves where the best continuous design is. Each material adds a base cost; titanium, at cost 0, gives the optimum |

## What the 0.5.0 results show

CMA-ES runs on the 21 problems without choice parameters. Against the genetic optimizer it wins 14, ties 3 (the integer spheres, which both solve) and loses 4.

- **Smooth and badly scaled problems: CMA-ES by far.** It solves sphere, ellipsoid and Rosenbrock to around 1e-15 in every dimension, where the genetic optimizer's median errors reach 4 on ellipsoid-10d and 7.4 on rosenbrock-10d. It also solves Ackley in every dimension, despite its many local minima.
- **Rastrigin in 5 and 10 dimensions: the genetic optimizer.** Its dense, regular grid of local minima traps CMA-ES even with restarts: a median error of 5.47 on rastrigin-10d against 0.228. The same holds for mixed-rastrigin-5d and -10d.
- **Categorical problems: only the genetic optimizer.** On material-selection-10d its median error is still exactly 1, the base cost of aluminium: most runs settle on the wrong material and never revisit the choice.
- **The genetic optimizer against random search:** it wins 22 problems, ties integer-sphere-2d, and still loses ellipsoid-2d (15.4 against 12.9), where its fixed mutation step is too coarse.

Since 0.5.0 every optimizer gets exactly the same number of evaluations. Before, the genetic optimizer used about 10% fewer than its budget, because it does not evaluate its elites again; its results are slightly better now for that reason alone.
