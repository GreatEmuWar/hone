# Changelog

## 0.9.0 (2026-10-31)

The genetic optimizer, retuned: adaptive mutation, no repeat evaluations and blend crossover. Against 0.8.2, its geometric-mean error on the benchmark's 36 problems falls to ×0.58, ×0.23, ×0.16, ×0.17 and ×0.012 at 10, 20, 50, 100 and 1,000 evaluations per dimension. Results of existing genetic optimizer runs change. Benchmark results are in [benchmarks/README.md](benchmarks/README.md).

### Added

- **Constraints.** `SearchSpace(parameters, constraints=[width * height <= 40, width >= 2 * height])` limits the search to points that satisfy every constraint, and the objective is never evaluated outside them. Constraints are written with range parameters and numbers, using `+`, `-`, `*`, `/`, `**` and `abs()`, compared with `<=` or `>=`. `Constraint.function(...)` covers anything else, such as choice parameters; it isn't saved with a result, so `rebuild_optimizer(functions=...)` takes it again. Initial points must satisfy the constraints, and constraints that leave nothing to search raise `InfeasibleSearchSpace` before any evaluation. Random search supports them so far; the other optimizers raise `UnsupportedSearchSpace` until they do.

### Changed

- Checkpoints no longer store the search space, which the optimizer loading one already has.

- **The genetic optimizer adapts its mutation.** Each individual carries its own mutation size, which its children inherit and vary, so selection tunes it as the run goes (`adaptive_mutation`, on by default). The default `mutation_rate` is now 1 over the number of variables, at least 0.1, rather than 0.1. On the benchmark's 36 problems the genetic optimizer's geometric-mean error falls to ×0.92, ×0.44, ×0.34, ×0.50 and ×0.33 at 10 to 1,000 evaluations per dimension, and it loses to random search on 14 problem and budget pairs rather than 37. With `noise=True` both stay as before, where they didn't help at large budgets. Results of existing runs change.
- **The genetic optimizer doesn't repeat evaluations.** Without noise, a child that would repeat a point already asked for, such as a copy of a parent that no mutation touched, is mutated again until it is new. About one child in four used to be such a copy. The geometric-mean error falls to ×0.90, ×0.82, ×0.91, ×0.89 and ×0.41 at 10 to 1,000 evaluations per dimension. This happens even with `mutation_rate=0`. With `noise=True` repeats stay, since they refine a point's average. Results of existing runs change.
- **The genetic optimizer blends its children's numbers.** Each range value of a child is drawn from between its parents' values, and up to `blend_alpha` (default 0.1) times their distance beyond, rather than copied from one parent (`blend_crossover`, on by default, with noise too). The geometric-mean error falls to ×0.69, ×0.62, ×0.51, ×0.38 and ×0.08 at 10 to 1,000 evaluations per dimension, and it loses to random search on 3 problem and budget pairs rather than 14. Results of existing runs change.
- **`choose_optimizer` sends problems with integer parameters to the genetic optimizer** above 200 evaluations, rather than CMA-ES, since with blend crossover it does better on them. Its pick's error on the benchmark is ×1.37 the best optimizer's, against ×1.62 had integers stayed with CMA-ES.

### Packaging

- **Releases go to PyPI.** Each new version is published as `hone-python` when it is released on GitHub, so `pip install hone-python` installs the latest release. This is the first.

## 0.8.2 (2026-10-31)

### Fixed

- **Elites could fill the whole population.** An `elitism_rate` below 1 could round to every individual, leaving nothing to evaluate, so the genetic optimizer stopped long before its budget was spent. At least one child is now bred each generation.

## 0.8.1 (2026-10-31)

### Fixed

- **Noisy runs stopped early at the threshold on a lucky value.** With `noise=True`, the threshold was checked against the best point's average as it stood when that point was first reported, not its current average. One lucky evaluation could end the run with `reached_threshold` set while the reported `best_fitness` had not reached `threshold`. The check now uses the reported best.

### Added

- **Property-based tests.** `pbt/tests/` checks the properties hone's documentation claims, using Hypothesis on generated search spaces and settings, and runs with the rest of the tests. `pbt/SUMMARY.md` lists them. Hypothesis is a new dev dependency.

## 0.8.0 (2026-10-31)

### Added

- **Choosing an optimizer automatically.** `choose_optimizer(search_space, max_evaluations, ...)` returns the optimizer the benchmark suggests: Bayesian optimization for 200 evaluations or fewer, then the genetic optimizer if there are choice parameters, and CMA-ES otherwise. On the benchmark, its pick's median error is 1.24 times the best optimizer's on geometric mean, against 5.9 times for always using CMA-ES.
- **Checkpoints.** `optimize(checkpoint="run.checkpoint")` saves the run's state after each generation, and carries the run on from the file if it exists, so a killed run picks up where it left off with the same result as an uninterrupted one. `save_checkpoint()` and `load_checkpoint()` do the same for ask and tell.
- **Not waiting for the slowest evaluations.** `min_finished` (default 1) is the share of each generation's evaluations that must finish before the next generation is bred. Below 1, slow evaluations finish in the background and their results join the generation under way, so workers no longer sit idle while one slow evaluation runs on. Bayesian optimization assumes evaluations still running come out as its model predicts, so it doesn't propose them again.

## 0.7.1 (2026-10-30)

### Added

- **A record of each run.** `Result.run` holds how the result was produced: the optimizer, its full config including the seed, the search space, any initial points, the objective's name, `workers` and `timeout`, the versions of hone, Python and numpy, and when the run started and how long it took.
- **Rebuilding an optimizer from a result.** `result.rebuild_optimizer()` sets up a new optimizer exactly as the result's was, so a saved run can be repeated.
- **Progress callbacks.** `optimize(callback=...)` is called after each generation with a `Progress`. Return `True` to stop the run. `print_progress` prints a line per generation.

### Changed

- **Ctrl+C returns the best result so far** rather than discarding the run. `optimize()` stops its workers and returns the result with `stopped` set. If no evaluation has succeeded yet, Ctrl+C still raises `KeyboardInterrupt`.
- **Result equality** compares what was found, not the new `run` record, so two runs with the same seed still compare equal.
- Worker processes ignore Ctrl+C; the main process stops them.

## 0.7.0 (2026-10-30)

### Added

- **Type information.** hone ships a `py.typed` marker, so type checkers and editors use its type hints in your code. CI checks hone's own types with mypy.

### Changed

- `SearchSpace` accepts any sequence of parameters, such as a tuple, not only a list.
- `Trial`, the candidate `ask()` hands out, is a pydantic model like the rest of hone's data, so `trial.model_dump()` gives it as a dict. Its fields are unchanged.

## 0.6.0 (2026-10-30)

New optimizers and parameter types, noise handling, time limits, and fixes from an expert review of 0.5.0. Benchmark results are in [benchmarks/README.md](benchmarks/README.md).

### Added

- **Bayesian optimization.** `BayesianOptimizer` models the objective with a Gaussian process and evaluates where it expects the most improvement. It is the best optimizer in the benchmark when evaluations are scarce: at 10 and 20 evaluations per dimension it has the lowest median error on 30 and 31 of the 36 problems. It supports range and choice parameters, but not `choose_n`.
- **Log-scale ranges.** `RangeParameter(..., log=True)` searches a range evenly in its logarithm, for settings that span orders of magnitude such as learning rates. It needs a positive minimum and works with `integer=True`.
- **Noisy objectives.** `noise=True` averages every evaluation of the same point, and each optimizer uses the averages. `final_evaluations` (off by default) keeps evaluations back to re-evaluate the most promising points at the end.
- **Starting points.** Every optimizer takes `initial_points`, such as a known good design or an earlier run's best parameters.
- **Time limits.** `optimize(timeout=...)` stops any evaluation that runs longer than that many seconds and counts it as failed. Evaluations now run in hone's own pool of worker processes, so a hung or crashed evaluation is replaced instead of stalling the run.
- **Niches for the genetic optimizer.** `niche_share` (off by default) breeds part of each generation within groups of parents that share the same choice parameter values. Set it to 0.5 when the best option of a choice parameter may be found late: on material-selection-10d it finds the right material in 10 of 10 runs, against 4 of 10 without.
- **Benchmarks.**
  - Every problem runs at five budgets, 10, 20, 50, 100 and 1,000 evaluations per dimension.
  - Rotated versions of four test functions, in which the variables interact: 36 problems in all.
  - `--ml` tunes three scikit-learn models by cross-validation, and `--noisy` adds noise to six problems.
  - `--jobs` shares the runs among processes.
- **An example**, `examples/tune_iris.py`, tuning a model on the Iris dataset.

### Changed

These change the results of existing code:

- **The genetic optimizer sizes its population from its budget.** With `max_evaluations` set and no `population_size`, the population is about a twentieth of the budget, between 16 and 50, and the run continues until the budget is spent. Without a budget the defaults are unchanged.
- **Tournaments pick their best entrant.** `tournament_p` now defaults to 1.0 rather than 0.75. The genetic optimizer's error falls by 11% to 64% across the benchmark's budgets.
- **CMA-ES with workers.** Unless `population_size` is set, `optimize(workers=n)` gives CMA-ES at least 2n candidates per generation, so its results depend on the number of workers as well as the seed.
- **Infinite values count as failed evaluations**, as NaN already did, and `tell()` rejects booleans.
- **An exception from the objective stops every worker at once**, rather than waiting for the evaluations already running.
- **Benchmark command line.** `--budget-per-dimension` is replaced by `--budgets`, and `run_benchmark` takes `budgets_per_dimension`.

### Fixed

- CMA-ES widened the steps of integer variables before learning from them, which inflated its step size on problems with integers. It now learns from the steps as drawn, as CMA-ES with Margin does.
- With the spawn and forkserver start methods, the default from Python 3.14, the time limit counted each worker's start-up, so on slow machines every evaluation could time out.

### Packaging

- The package is named `hone-python`, because `hone` is taken on PyPI: `pip install hone-python`. It still imports as `hone`.

### Dependencies

- `scipy` and `threadpoolctl` are now required. `scikit-learn` is needed only for the machine learning benchmark.

## 0.5.0

A rewrite of the 2023 version.

- Parameters and configs are validated pydantic models: `RangeParameter`, `ChoiceParameter` and `FixedParameter` replace `Constraint`.
- Every optimizer shares a base class with an ask/tell interface, parallel workers, failed evaluations, an exact `max_evaluations` budget, seeds and a savable `Result`.
- CMA-ES, alongside the genetic optimizer and random search.
- Fixes to the genetic optimizer's loop, selection, mutation and result.
- A benchmark suite with a regression check in CI.
- Python 3.12 or later; licensed under BSD-3-Clause.
