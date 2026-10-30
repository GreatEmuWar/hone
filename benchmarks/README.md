# Benchmarks

This folder holds the results of `python -m hone.benchmarks`:

| File | Contents |
| --- | --- |
| [results.md](results.md) | The full suite's results table, from `make benchmark` |
| `full-summaries.json` | The same results as data |
| [noisy-results.md](noisy-results.md) | The noisy problems' results, from `make benchmark-noisy` |
| `noisy-summaries.json` | The same results as data |
| [ml-results.md](ml-results.md) | The machine learning tuning results, from `make benchmark-ml` |
| `ml-summaries.json` | The same results as data |
| `quick-baseline.json` | The quick suite's results, which CI compares every change against |

## What the suite measures

Each optimizer runs on each problem with 10 seeds at five budgets: 10, 20, 50, 100 and 1,000 evaluations per dimension, so a 5-dimensional problem gets 50, 100, 250, 500 and 5,000 evaluations. Each budget is a separate run. The small budgets stand for expensive evaluations, such as a simulation that takes minutes, where a few dozen runs may be all there is time for. At every budget all optimizers get exactly the same number of evaluations.

The results start with an overview: for each budget, the number of problems on which each optimizer has the lowest median error. A table per budget follows, reporting the median **error**, the best value found minus the problem's known optimum, and the share of runs **solved** to an error of 0.0001 or less.

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

### At small budgets

- **Bayesian optimization is the best optimizer when evaluations are scarce.** At 10 and 20 evaluations per dimension it has the lowest median error on 30 and 31 of the 36 problems, and it beats CMA-ES on 28 and 29 of the 33 they share. On sphere-10d with 100 evaluations its median error is 0.0058, against 8.5 for CMA-ES; on ellipsoid-10d it is 666 against 235,000. It is also the only optimizer that reliably picks the right material at small budgets: on material-selection-5d at 20 evaluations per dimension its median error is 0.00066, against 2.43 for the genetic optimizer. Its losses are mostly rotated Rosenbrock, a long curved valley its model struggles to follow, and Rastrigin in 2 dimensions.
- **It runs only up to 200 evaluations.** Each candidate costs a model fit and a search over the model, so the benchmark leaves it out of larger budgets (marked n/a): in the full suite it runs at 50 evaluations per dimension only on the 2-dimensional problems, and not at 100 or 1,000.
- **Without Bayesian optimization, CMA-ES would be the best at every small budget.** With 50 evaluations per dimension on sphere-10d its median error is 0.0245, against 1.78 for the genetic optimizer and 21.8 for random search.
- **The genetic optimizer sizes its population from the budget.** With a fixed population of 50, a budget of 50 evaluations or fewer ran out during its first, random generation, so it was exactly random search on every problem with 5 dimensions or fewer. It now uses about a twentieth of the budget, between 16 and 50. Against the fixed population, its geometric-mean error falls by 8% at 10 evaluations per dimension, 16% at 20, 28% at 50 and 12% at 100, and is unchanged at 1,000. It now beats random search on 20 problems at 10 evaluations per dimension and 31 at 50, up from 9 and 29.
- **The 2-dimensional ellipsoid and Rosenbrock got worse.** Smaller populations converge sooner, and on these badly scaled problems the fixed mutation step can't follow. ellipsoid-2d at 50 evaluations per dimension is about 11 times worse. These are problems on which the genetic optimizer already lost to random search, and an adaptive mutation step is the planned fix.
- **Categorical problems have no strong option yet.** CMA-ES can't run them. The smaller population helps the genetic optimizer on material-selection (median error 2.79 rather than 6.17 in 10 dimensions at 50 evaluations per dimension), but it still often settles on the wrong material.

### At 1,000 evaluations per dimension

CMA-ES runs on the 33 problems without choice parameters. Against the genetic optimizer it wins 26, ties 3 (the integer spheres, which both solve) and loses 4: plain Rastrigin in 5 and 10 dimensions and plain mixed Rastrigin in 5 and 10 dimensions.

- **Those four wins come from separability.** Uniform crossover takes each variable from one parent or the other, which works well when variables act independently. Rotated, the genetic optimizer's median error on Rastrigin rises from 0.228 to 22.4 in 10 dimensions, about a hundredfold, while CMA-ES's barely changes (5.47 to 6.47). CMA-ES wins all 12 rotated problems.
- **Smooth and badly scaled problems: CMA-ES by far, rotated or not.** It solves sphere, ellipsoid, Rosenbrock and Ackley to between about 1e-11 and 1e-17 in every dimension. On rotated-ellipsoid-10d the genetic optimizer's median error is 7,050.
- **Rastrigin stays hard for CMA-ES at this budget.** Its deep, dense local minima trap it; solving it reliably takes much larger populations than 1,000 evaluations per dimension allows the restarts to reach.
- **Categorical problems: only the genetic optimizer.** On material-selection-10d its median error is exactly 1, the base cost of aluminium: most runs settle on the wrong material.
- **The genetic optimizer against random search:** it loses ellipsoid-2d, rotated-ellipsoid-2d, rotated-rosenbrock-2d and rotated-rastrigin-2d, where its fixed mutation step is too coarse for the precision those problems need.

Since 0.5.0 every optimizer gets exactly the same number of evaluations. Before, the genetic optimizer used about 10% fewer than its budget, because it does not evaluate its elites again.

## Machine learning tuning

`python -m hone.benchmarks --ml` tunes three scikit-learn models by 5-fold cross-validation, at 10, 20 and 50 evaluations per dimension: a support vector classifier on breast cancer data (C and gamma), kernel ridge regression on diabetes data (alpha, gamma and the kernel), and a random forest on wine data (depth, leaf size, features per split and split criterion). Each problem minimizes 1 − accuracy or 1 − R², so an error of 0.0035 on the breast cancer problem is 0.35 percentage points of accuracy, about 2 of its 569 samples.

The optimum of a real model is unknown, so error is measured from the best score of a reference search: Bayesian optimization with 200 evaluations, and CMA-ES (where it applies), the genetic optimizer and random search with 1,000 each, three seeds of each.

What the results show:

- **Bayesian optimization is the best tuner.** At 20 and 50 evaluations per dimension it has the lowest median error on all three problems. On the kernel ridge problem it finds the reference R² in 60% of runs with 60 evaluations and in every run with 150, where no other optimizer gets within 0.0005 in its median run.
- **The differences are smaller than on the test functions.** Most settings of a well-chosen model score within a percentage point or two of the best, so random search is a fair tuner, as the hyperparameter tuning literature also finds. The extra value of a better optimizer shows in the last fraction of a point, and in how few evaluations it takes to get there.
- **Scores move in steps.** Accuracy changes in multiples of one sample, so several optimizers often stop at exactly the same score: the random forest's median error is 0.0057 for every optimizer at 10 evaluations per dimension.
- **The genetic optimizer is no better than random search here.** It ties or loses on the SVM problem at every budget, and on kernel ridge it wins only at 50 evaluations per dimension. With budgets of 20 to 200 evaluations it runs only a few generations.

## Noisy problems

`python -m hone.benchmarks --noisy` adds Gaussian noise with a standard deviation of 1 to every evaluation of six 5-dimensional problems: sphere, Rosenbrock, Rastrigin, rotated Ackley, mixed Rastrigin and material selection. Near their optima these problems' values differ by less than that, so the noise hides the differences an optimizer has to find. Every optimizer runs with `noise=True`, and each run is scored by the true, noise-free value of the point it returns, so a lucky result doesn't count. The noise is repeatable for each seed.

- **Bayesian optimization is best at small budgets:** it has the lowest median error on all six problems at 10 evaluations per dimension, and on noisy-sphere-5d its median error is 0.44 against 3.3 for CMA-ES.
- **At 200 evaluations per dimension, CMA-ES and the genetic optimizer share the wins**, three problems each.

### What `noise=True` does, measured

Each design choice was compared with and without it on these problems, over 10 seeds, as the ratio of geometric-mean errors:

| Change | 10 / dim | 40 / dim | 200 / dim |
| --- | ---: | ---: | ---: |
| Genetic optimizer: averaging, with parents ranked by their latest average | ×1.00 | ×0.85 | ×0.80 |
| Bayesian optimization: improvement measured against the model's best prediction | ×0.95 | ×0.90 | not run |
| CMA-ES: 10% of the budget kept back for a final round | ×1.12 | ×1.14 | ×0.89 |
| Random search: the same | ×1.02 | ×1.02 | ×0.98 |

Some ideas that sound right made things worse, and were left out:

- **Re-evaluating the genetic optimizer's elites every generation** cost evaluations and gave up most of the gain from averaging (×1.08, ×0.99 and ×1.04).
- **Telling Bayesian optimization's model to expect noise**, through a prior on the noise level, made it worse than no noise handling at all (×1.13 and ×1.40); the model learns the noise from the data well enough.
- **Choosing Bayesian optimization's result by the model's prediction** rather than by the averages was also worse than no noise handling (×1.13 at both budgets).
- **The final round** costs more search than it gains until errors approach the noise level, which is why it is off by default.
