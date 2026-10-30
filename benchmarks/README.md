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

## What the 0.3.0 results show

- The genetic optimizer beats random search on 22 of the 24 problems, often by 100 to 10,000 times in 5 and 10 dimensions. They tie on integer-sphere-2d, which is small enough for random search to find the exact point.
- On ellipsoid-2d it does worse than random search. That function needs one variable set to within about 0.004, and the fixed mutation step of 10% of the range is too coarse. An adaptive step size is the planned fix.
- On material-selection-10d its median error is exactly 1, the base cost of aluminium: most runs settle on the wrong material and never revisit the choice.
- Few runs reach the 0.0001 tolerance within the budget. That is the gap stronger optimizers such as CMA-ES are expected to close.
