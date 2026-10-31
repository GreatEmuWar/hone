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
| [constrained-results.md](constrained-results.md) | The constrained problems' results, from `make benchmark-constrained` |
| `constrained-summaries.json` | The same results as data |
| [conditional-results.md](conditional-results.md) | The conditional problems' results, from `make benchmark-conditional` |
| `conditional-summaries.json` | The same results as data |
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

## What the results show

### At small budgets

- **Bayesian optimization is the best optimizer when evaluations are scarce.** At 10 and 20 evaluations per dimension it has the lowest median error on 29 and 30 of the 36 problems, and it beats CMA-ES on 28 and 29 of the 33 they share. On sphere-10d with 100 evaluations its median error is 0.0058, against 8.5 for CMA-ES; on ellipsoid-10d it is 666 against 235,000. It is also the only optimizer that reliably picks the right material at small budgets: on material-selection-5d at 20 evaluations per dimension its median error is 0.00066, against 1.7 for the genetic optimizer. Its losses are mostly rotated Rosenbrock, a long curved valley its model struggles to follow, and Rastrigin in 2 dimensions.
- **It runs only up to 200 evaluations.** Each candidate costs a model fit and a search over the model, so the benchmark leaves it out of larger budgets (marked n/a): in the full suite it runs at 50 evaluations per dimension only on the 2-dimensional problems, and not at 100 or 1,000.
- **Without Bayesian optimization, CMA-ES would be the best at every small budget.** With 50 evaluations per dimension on sphere-10d its median error is 0.0245, against 0.10 for the genetic optimizer and 21.8 for random search.
- **The genetic optimizer sizes its population from the budget.** With a fixed population of 50, a budget of 50 evaluations or fewer ran out during its first, random generation, so it was exactly random search on every problem with 5 dimensions or fewer. It now uses about a twentieth of the budget, between 16 and 50. Against the fixed population, its geometric-mean error falls by 8% at 10 evaluations per dimension, 16% at 20, 28% at 50 and 12% at 100, and is unchanged at 1,000. With the tournament fix, adaptive mutation, repeat avoidance and blend crossover below, it beats random search on 33 problems at 10 evaluations per dimension and all 36 at 50, up from 9 and 29 before these changes.
- **Tournaments now always pick their best entrant.** They used to pick the worst of the four 25% of the time (`tournament_p=0.75`), which slowed the search for nothing. With `tournament_p=1.0` the genetic optimizer's geometric-mean error falls by 11% at 10 evaluations per dimension, 17% at 20, 64% at 50, 21% at 100 and 24% at 1,000, and it improves on 22 to 28 of the 36 problems at each budget. Tournament sizes of 2, 3, 6 and 8 were also tried; 4 stays.
- **Mutation adapts as the run goes.** Each individual carries its own mutation size, which its children inherit and vary, so selection tunes it to the problem's current scale; and each value mutates with probability 1 over the number of variables, at least 0.1, rather than always 0.1. The genetic optimizer's geometric-mean error falls to ×0.92 at 10 evaluations per dimension, ×0.44 at 20, ×0.34 at 50, ×0.50 at 100 and ×0.33 at 1,000. It used to lose to random search on 37 problem and budget pairs, mostly badly scaled ones its fixed step couldn't follow; it now loses on 14, and on none at 1,000 evaluations per dimension. ellipsoid-2d at 50 evaluations per dimension falls from 11,600 to 74, against 867 for random search. Changing the step from the parents' spread instead, or letting it adapt faster, did worse. With `noise=True` both changes are off, because on the noisy problems they cost more at 200 evaluations per dimension than they gained at small budgets; noisy results are unchanged.
- **Children no longer repeat points already evaluated.** About one child in four used to be an exact copy of a point already asked for, most often a parent that no mutation touched, and evaluating it again only spent the budget. Such a child is now mutated again, one value at a time, until it is new. The genetic optimizer's geometric-mean error falls to ×0.90 at 10 evaluations per dimension, ×0.82 at 20, ×0.91 at 50, ×0.89 at 100 and ×0.41 at 1,000, and every problem family improves. Doubling the mutation size on each retry changed nothing measurable. With `noise=True` repeats stay, because they refine a point's average; noisy results are unchanged.
- **Children's numbers are blended from their parents'.** Uniform crossover copied each number from one parent or the other, so a child could only recombine values the population already had. Blend crossover draws each range value from between the two parents' values, and up to a tenth of their distance beyond either. The genetic optimizer's geometric-mean error falls to ×0.69 at 10 evaluations per dimension, ×0.62 at 20, ×0.51 at 50, ×0.38 at 100 and ×0.08 at 1,000; it is better on 136 problem and budget pairs and worse on 17, and loses to random search on 3 rather than 14. Widths of 0, 0.05 and 0.2 did about as well as 0.1; 0.3 and 0.5 did worse. With blend crossover, mutation sizes stop shrinking, since crossover does the fine convergence, but adaptive mutation still earns its place: without it the error at 1,000 evaluations per dimension is 2.3 times higher. Blend crossover also stays on with `noise=True`, where it helped (×0.67, ×0.84 and ×0.99 on the noisy problems).
- **Categorical problems have no strong option yet.** CMA-ES can't run them. The genetic optimizer still often settles on the wrong material: its median error on material-selection-10d at 50 evaluations per dimension is 2.1. The opt-in niches below fix that on larger budgets.

### At 1,000 evaluations per dimension

CMA-ES runs on the 33 problems without choice parameters. Against the genetic optimizer it wins 24, ties 4 (the integer spheres and mixed-rastrigin-2d, which both solve) and loses 5: plain Rastrigin and plain mixed Rastrigin in 5 and 10 dimensions, and rotated Rastrigin in 2.

- **Most of those wins come from separability.** Uniform crossover takes each variable from one parent or the other, which works well when variables act independently. Rotated, the genetic optimizer's median error on Rastrigin rises from 0.0014 to 9.5 in 10 dimensions, nearly seven-thousandfold, while CMA-ES's barely changes (5.47 to 6.47). CMA-ES wins 11 of the 12 rotated problems.
- **Smooth and badly scaled problems: CMA-ES by far, rotated or not.** It solves sphere, ellipsoid, Rosenbrock and Ackley to between about 1e-11 and 1e-17 in every dimension. On rotated-ellipsoid-10d the genetic optimizer's median error is 1,600.
- **Rastrigin stays hard for CMA-ES at this budget.** Its deep, dense local minima trap it; solving it reliably takes much larger populations than 1,000 evaluations per dimension allows the restarts to reach.
- **Categorical problems: only the genetic optimizer.** On material-selection-10d 6 of its 10 runs find titanium and solve it, but the other 4 settle on a wrong material with a base cost of 1 to 3. With `niche_share=0.5`, which breeds half of each generation within groups sharing the same material, all 10 runs find titanium. At 200 evaluations per dimension niches raise the runs that find titanium from 6 to 9 of 10, though the median error rises from 0.012 to 0.029, because tuning the numbers within each niche is slower. Niches cost a little on small budgets (material-selection-5d at 50 evaluations per dimension gets worse), so they are off by default, and they change nothing on problems without choice parameters.
- **The genetic optimizer against random search:** it now beats or ties it on every problem. Before adaptive mutation it lost five, among them rosenbrock-2d and rotated-ellipsoid-2d, where its fixed mutation step was too coarse for the precision they need.

CMA-ES now widens the steps of integer variables only where they are sampled, and adapts from the unwidened steps, as CMA-ES with Margin does. Before, the widening also inflated the steps of the continuous variables in the same run. This changes nothing measurable on the full suite, where integer variables are few, but on a 10-dimensional problem with one integer variable it lets the continuous variables converge as they would without it.

Since 0.5.0 every optimizer gets exactly the same number of evaluations. Before, the genetic optimizer used about 10% fewer than its budget, because it does not evaluate its elites again.

## Choosing an optimizer automatically

`choose_optimizer` turns these results into a rule: Bayesian optimization for 200 evaluations or fewer, then the genetic optimizer if there are choice or integer parameters, and CMA-ES otherwise. Integer problems went to CMA-ES until blend crossover made the genetic optimizer better on them; sending them to it lowers the pick's error on the standard suite from ×1.62 to ×1.37. For each problem and budget, its pick's median error was compared with the best optimizer's:

| Suite | Pick's error against the best, geometric mean | Best on | Always CMA-ES | Always the genetic optimizer |
| --- | ---: | ---: | ---: | ---: |
| Standard, 36 problems × 5 budgets | ×1.37 | 146 of 180 | ×7.9, and 15 it can't run | ×39 |
| Noisy, 6 × 3 | ×1.07 | 12 of 18 | ×1.69, and 3 it can't run | ×1.58 |
| Machine learning, 3 × 3 | ×1.05 | 8 of 9 | ×1.14, and 6 it can't run | ×23 |
| Constrained, 10 × 5 | ×1.04 | 45 of 50 | ×21 | ×202 |

A test (`tests/test_choose_optimizer.py`) recomputes this from the saved summaries, so the rule stays backed by the results whenever they are regenerated. A rule that also limited Bayesian optimization to 50 evaluations per variable did worse (×1.36 on the standard suite), so the simpler one stays.

## Machine learning tuning

`python -m hone.benchmarks --ml` tunes three scikit-learn models by 5-fold cross-validation, at 10, 20 and 50 evaluations per dimension: a support vector classifier on breast cancer data (C and gamma), kernel ridge regression on diabetes data (alpha, gamma and the kernel), and a random forest on wine data (depth, leaf size, features per split and split criterion). Each problem minimizes 1 − accuracy or 1 − R², so an error of 0.0035 on the breast cancer problem is 0.35 percentage points of accuracy, about 2 of its 569 samples.

The optimum of a real model is unknown, so error is measured from the best score of a reference search: Bayesian optimization with 200 evaluations, and CMA-ES (where it applies), the genetic optimizer and random search with 1,000 each, three seeds of each.

What the results show:

- **Bayesian optimization is the best tuner.** At 20 and 50 evaluations per dimension it has the lowest median error on all three problems. On the kernel ridge problem it finds the reference R² in 60% of runs with 60 evaluations and in every run with 150, where no other optimizer gets within 0.0005 in its median run.
- **The differences are smaller than on the test functions.** Most settings of a well-chosen model score within a percentage point or two of the best, so random search is a fair tuner, as the hyperparameter tuning literature also finds. The extra value of a better optimizer shows in the last fraction of a point, and in how few evaluations it takes to get there.
- **Scores move in steps.** Accuracy changes in multiples of one sample, so several optimizers often stop at exactly the same score: the random forest's median error is 0.0057 for every optimizer at 10 evaluations per dimension.
- **The genetic optimizer is better than random search here, but not by much.** It wins on kernel ridge at 10 and 20 evaluations per dimension and on the SVM at 20 and 50, ties on the SVM at 10 and on the random forest, and loses narrowly on kernel ridge at 50. Adaptive mutation cut its median error to ×0.74 to ×0.84; avoiding repeats changed it by ×1.15, ×0.84 and ×0.83, and blend crossover by ×0.91, ×0.97 and ×1.47, at 10, 20 and 50 evaluations per dimension. Both losses are on kernel ridge. With budgets of 20 to 200 evaluations it runs only a few generations.

## Conditional problems

`python -m hone.benchmarks --conditional` runs four problems with conditional parameters, at the same five budgets, each twice: with its conditions, and as a `-flat` version with every parameter always on and the function ignoring those it doesn't use. The flat version is how such a space was searched before conditions, so the pair shows what conditions are worth.

| Problem | What it tests |
| --- | --- |
| branches-3x2d and branches-3x5d | A choice of three branches, each with its own 2 or 5 numbers. The best is a steep, narrow bowl reaching 0; the decoys are a wide, shallow bowl with a floor of 1 and a shallow Rastrigin grid with a floor of 0.5, where random points score better |
| nested-9 | A choice of model, one with a choice of kernel, one kernel with numbers of its own: the best point is two conditions deep, past a wide decoy with a floor of 0.25 |
| layers-7 | A layer count from 1 to 5 whose larger values switch on more numbers; 3 layers is best. Its conditions test a range, so CMA-ES can search it |

Geometric-mean error over the four problems, with conditions over the flat versions:

| Optimizer | 10 / dim | 20 / dim | 50 / dim | 100 / dim | 1,000 / dim |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bayesian optimization | ×0.0004 | ×0.0003 | not run | not run | not run |
| Genetic optimizer | ×1.04 | ×0.86 | ×0.05 | ×0.08 | ×0.00004 |
| CMA-ES (layers-7 only) | ×1 | ×1 | ×1 | ×1 | ×1 |
| Random search | ×1 | ×1 | ×1 | ×1 | ×1 |

- **Bayesian optimization gains the most.** Points that differ only in a parameter switched off are the same point to its model, so the values those parameters happen to hold add no noise. On branches-3x2d at 10 evaluations per dimension its median error is 0.00001 with conditions, against 0.87 flat; on nested-9, 0.0006 against 0.48. It is the best optimizer on most problems at 10 and 20 evaluations per dimension.
- **The genetic optimizer gains from niches, which conditions turn on.** Without niches it settled in a decoy branch on most problems, flat or not: a wide branch fills the population before the narrow best one is tuned. With conditions, `niche_share` defaults to 0.5, so each branch breeds within itself, and at 1,000 evaluations per dimension it solves every run of the three branch problems. With niches set on the flat versions too, conditions still help at 50 and 1,000 evaluations per dimension (×0.09 and ×0.23) but not at 100 (×3.1), where layers-7 decides it: its runs either find three layers or settle on two or four, which costs 0.5, so its median flips between budgets.
- **CMA-ES and random search come out the same** with conditions as without: they draw the same points, and an objective that ignores the parameters switched off sees the same values.
- **`choose_optimizer`'s rule holds without change.** Its pick is 1.15 times the best optimizer's error on geometric mean over all eight problems, and 1.07 times over the four with conditions.

The machine learning suite has a conditional problem too: model-selection-breast-cancer, choosing a support vector classifier, a random forest or k nearest neighbours and that model's settings (see [Machine learning tuning](#machine-learning-tuning)).

## Constrained problems

`python -m hone.benchmarks --constrained` runs ten problems whose optimum lies on a constraint's boundary, as it usually does in engineering, at the same five budgets:

| Problem | What it tests |
| --- | --- |
| boundary-sphere and boundary-ellipsoid, in 2, 5 and 10 dimensions | A linear constraint that cuts off the unconstrained optimum, so the answer is on the cutting plane; the optimum is known exactly |
| rosenbrock-disk-2d | Rosenbrock's curved valley inside the disk x² + y² ≤ 2, with the optimum (1, 1) on the disk's edge |
| pressure-vessel | A vessel of least cost holding at least 1,296,000 cubic inches, with plate thicknesses in sixteenths of an inch: integers and three constraints |
| tension-spring | A spring of least weight under four limits; only 0.8% of the box satisfies them all |
| welded-beam | A welded beam of least cost under six limits, two of them functions with square roots; 2.9% of the box is feasible |

The engineering problems are scored against the best designs published for them. Every optimizer only ever asks for feasible points, so each evaluation counts.

- **Bayesian optimization is the best at small budgets**, more clearly than without constraints: it has the lowest median error on all ten problems at 10 and 20 evaluations per dimension, and all six it runs on at 50. On the pressure vessel with 40 evaluations its median cost is 266 above the best known design, against 10,010 for CMA-ES and 19,140 for random search.
- **CMA-ES is the best at 1,000 evaluations per dimension**, on all ten problems. It finds the best known design of nine of them in every run, and of the pressure vessel in three of ten. When a candidate breaks a constraint, it draws again from the same distribution. Ranking such candidates below every feasible one without evaluating them was also tried, and did worse at every budget (×1.17 to ×1.59).
- **The genetic optimizer is behind CMA-ES under constraints**, but it is the optimizer for constrained problems with choice or integer parameters, which CMA-ES can't search or handles less well. A child that breaks a constraint is moved back toward its fitter parent, halving the gap ten times, so it lands near the boundary. Against breeding a new child instead, this cut its geometric-mean error to ×0.84, ×0.68, ×0.74, ×0.46 and ×0.44 at 10 to 1,000 evaluations per dimension; on the welded beam at 1,000 it went from 0.080 to 0.0058. It beats random search on every problem at 50 and 1,000 evaluations per dimension, and on nine of ten at 100, losing on rosenbrock-disk-2d.
- **`choose_optimizer`'s rule holds without change.** Its pick is 1.04 times the best optimizer's error on geometric mean, and the best on 45 of the 50 problem and budget pairs.

## Noisy problems

`python -m hone.benchmarks --noisy` adds Gaussian noise with a standard deviation of 1 to every evaluation of six 5-dimensional problems: sphere, Rosenbrock, Rastrigin, rotated Ackley, mixed Rastrigin and material selection. Near their optima these problems' values differ by less than that, so the noise hides the differences an optimizer has to find. Every optimizer runs with `noise=True`, and each run is scored by the true, noise-free value of the point it returns, so a lucky result doesn't count. The noise is repeatable for each seed.

- **Bayesian optimization is best at small budgets:** it has the lowest median error on all six problems at 10 evaluations per dimension, and on noisy-sphere-5d its median error is 0.44 against 3.3 for CMA-ES.
- **At 200 evaluations per dimension, the genetic optimizer has the most wins**, five problems against one for CMA-ES.

### What `noise=True` does, measured

Each design choice was compared with and without it on these problems, over 10 seeds, as the ratio of geometric-mean errors:

| Change | 10 / dim | 40 / dim | 200 / dim |
| --- | ---: | ---: | ---: |
| Genetic optimizer: averaging, with parents ranked by their latest average | ×1.01 | ×0.92 | ×0.74 |
| Bayesian optimization: improvement measured against the model's best prediction | ×0.95 | ×0.90 | not run |
| CMA-ES: 10% of the budget kept back for a final round | ×1.12 | ×1.14 | ×0.89 |
| Random search: the same | ×1.02 | ×1.02 | ×0.98 |

Some ideas that sound right made things worse, and were left out:

- **Re-evaluating the genetic optimizer's elites every generation** cost evaluations and gave up most of the gain from averaging (×1.08, ×0.99 and ×1.04).
- **Telling Bayesian optimization's model to expect noise**, through a prior on the noise level, made it worse than no noise handling at all (×1.13 and ×1.40); the model learns the noise from the data well enough.
- **Choosing Bayesian optimization's result by the model's prediction** rather than by the averages was also worse than no noise handling (×1.13 at both budgets).
- **The final round** costs more search than it gains until errors approach the noise level, which is why it is off by default.
