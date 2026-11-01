# Changelog

## Unreleased

### Added

- **A documentation site**, at https://greatemuwar.github.io/hone/, built with MkDocs Material. Its guide, examples, benchmark results and changelog are pulled from the README, the example scripts and the benchmark files, so each is still written once, and its API reference is generated from the docstrings. Every config field now has a docstring, which also becomes its description in the config's JSON schema. CI builds the site on every pull request and publishes it on every push to `main`; see "Docs" in the README.

### Internal

- **CI is hardened against supply-chain attacks.** Every workflow is read-only unless a job needs more, every action is pinned by commit, and checkouts don't keep the token. Poetry is installed from a hash-pinned requirements file, and the Docker image's base is pinned by digest and runs the tests as an ordinary user. Released packages carry a build provenance attestation, which `gh attestation verify` checks, alongside PyPI's own. Pull requests are checked for dependencies with known vulnerabilities, zizmor lints the workflows, and OpenSSF Scorecard scores the repository weekly. Dependabot keeps actions, dependencies, Poetry and the base image current.

## 0.12.1 (2026-10-31)

### Added

- **A security policy**, [SECURITY.md](SECURITY.md): how to report a vulnerability privately through GitHub, which versions get fixes (the latest minor release), and what counts. Running your own objective and constraint functions and loading pickled checkpoints are by design; running code or touching files when loading a result or a config is a vulnerability. The README links to it.

## 0.12.0 (2026-10-31)

Several objectives at once: a run returns the front of best trade-offs rather than one answer to a weighted score, and the choice between them is made after seeing what each costs. NSGA2 searches for the front on larger budgets and multi-objective Bayesian optimization on small ones, `choose_pareto_optimizer` picks between them, and a multi-objective benchmark suite backs the choice. A README section explains them, and an example designs a beam that is light, cheap and stiff. See [benchmarks/README.md](benchmarks/README.md#multi-objective-problems).

### Added

- **Several objectives at once.** `MultiObjectiveConfig(objectives={"accuracy": "maximize", "cost": "minimize"})` names each objective, and the objective function returns a dict of them. A run returns a `ParetoResult` whose `front` holds the best trade-offs among everything evaluated: `front.best("accuracy", where={"cost": 20})` is the most accurate design costing at most 20, `front.closest()` the most balanced by augmented Chebyshev distance on the front's own scale, and `front.hypervolume(reference)` scores the front. With noise, each point on the front is an average and comes with its standard error. A dict with the wrong keys raises `TypeError`. `MultiObjectiveRandomSearch` is the baseline, and `print_progress` prints the size of the front and its hypervolume. See [docs/design/multiple-objectives.md](docs/design/multiple-objectives.md).
- **NSGA2**, a genetic algorithm that searches for the front: it ranks the population by fronts and spreads it along them by crowding distance, and breeds with the genetic optimizer's crossover, mutation, constraint repair and niches, so it supports every kind of parameter, constraints and conditions. At 1,000 evaluations per dimension its median hypervolume gap is about 0.001 on the ZDT problems, ten times closer than weighted sums.
- **MultiObjectiveBayesian**, for budgets of a few hundred evaluations or fewer. Each candidate draws random weights for the objectives and goes where a Gaussian process expects the most improvement in their augmented Chebyshev combination. `method="parego"`, the default, models the combination; `method="models"` models each objective, which did better on the three-objective problem. At 10 evaluations per dimension ParEGO has the lowest gap on every problem any optimizer reaches.
- **`choose_pareto_optimizer`** picks MultiObjectiveBayesian up to 200 evaluations and NSGA2 beyond, as the benchmark suggests.
- **A multi-objective benchmark suite.** `python -m hone.benchmarks --multi` runs ZDT1 (also rotated), ZDT2, ZDT3, ZDT4, DTLZ2 and the welded beam's cost against its deflection, scored by hypervolume gap from the true front, against weighted sums of the objectives as the baseline. The summaries include each run's median attainment surface. See [benchmarks/README.md](benchmarks/README.md#multi-objective-problems).
- **An example**, `examples/cantilever_beam.py`, designing a cantilever beam in steel, aluminium or carbon fibre against its mass, cost and deflection, under a stress limit that depends on the material, and choosing from the front.

### Internal

- The optimizer base class is split into `OptimizerCore`, the generation loop, ask and tell, workers, checkpoints and run records shared by every optimizer, and `Optimizer`, which adds what a single objective needs. This makes room for multiple objectives (see [docs/design/multiple-objectives.md](docs/design/multiple-objectives.md)). Results don't change, and checkpoints saved by 0.11.0 still load.

## 0.11.0 (2026-10-31)

Conditional parameters: parameters that exist only while others take certain values, written `model["svm"] >> [kernel, C]`, which every optimizer supports. A README section explains them, an example chooses a classifier and its settings in one search, and the benchmark has a conditional suite comparing each optimizer with searching every parameter all the time. See [benchmarks/README.md](benchmarks/README.md#conditional-problems).

### Added

- **Conditional parameters.** `SearchSpace(parameters, conditions=[model["svm"] >> [kernel, C], kernel["rbf"] >> gamma])` switches parameters on only while a test holds: a choice parameter's values, such as `model["svm"]`, or a comparison of range parameters in brackets, such as `(layers >= 3) >> units_3`. A parameter that is switched off isn't passed to the objective, which gives it a default, and results and initial points leave it out. Constraints on it apply only while it is switched on, and `test >> constraint` makes a constraint apply only while the test holds. Every optimizer supports them. See [docs/design/conditions.md](docs/design/conditions.md).
  - Bayesian optimization encodes a parameter switched off the same whatever value it holds, so its model isn't misled by values that don't matter. On the conditional benchmark its geometric-mean error is ×0.0004 and ×0.0003 that of the same problems with every parameter always on, at 10 and 20 evaluations per dimension.
  - The genetic optimizer keeps a value for every gene, switched off or not, takes a gene from the parent it is on in, and mutates only genes switched on. With conditions, `niche_share` defaults to 0.5, so each branch breeds within itself; its error is ×0.05, ×0.08 and ×0.00004 the always-on version's at 50, 100 and 1,000 evaluations per dimension.
  - CMA-ES searches conditions on range parameters, keeping a parameter's place in its distribution while it is switched off.
- **A conditional benchmark suite.** `python -m hone.benchmarks --conditional` runs four problems with conditional parameters, each also as a flat version with every parameter always on, to compare. The machine learning suite gains model-selection-breast-cancer, which chooses a classifier and its settings, and its flat version.
- **An example**, `examples/choose_a_model.py`, choosing among an SVM with three kernels, a random forest and logistic regression, with each one's settings and a conditional constraint on the forest's size.

### Changed

- `GeneticOptimizerConfig.niche_share` defaults to `None`: 0.5 for a search space with conditions, and 0, as before, otherwise. Results without conditions don't change.

## 0.10.0 (2026-10-31)

Constraints: limits across parameters, written with the parameters themselves, which every optimizer keeps to. A README section explains them, two examples work through a standard engineering design and a maintenance schedule, and the benchmark has a constrained suite. See [benchmarks/README.md](benchmarks/README.md#constrained-problems).

### Added

- **Constraints.** `SearchSpace(parameters, constraints=[width * height <= 40, width >= 2 * height])` limits the search to points that satisfy every constraint, and the objective is never evaluated outside them. Constraints are written with range parameters and numbers, using `+`, `-`, `*`, `/`, `**` and `abs()`, compared with `<=` or `>=`. `Constraint.function(...)` covers anything else, such as choice parameters; it isn't saved with a result, so `rebuild_optimizer(functions=...)` takes it again. Initial points must satisfy the constraints, and constraints that leave nothing to search raise `InfeasibleSearchSpace` before any evaluation. Every optimizer supports them: random search and Bayesian optimization only consider feasible points, CMA-ES draws a candidate again while it breaks a constraint, and the genetic optimizer moves an infeasible child back toward its fitter parent until it is feasible, near the boundary. `python -m hone.benchmarks --constrained` runs ten constrained problems, among them the pressure vessel, tension spring and welded beam design problems.
- **Two worked examples.** `examples/pressure_vessel.py` designs a pressure vessel of least cost under three constraints, with integer plate thicknesses, and finds the best published design. `examples/maintenance_schedule.py` chooses a production line's maintenance schedule, scored by a noisy simulation of a year, under a limit on technician hours and one on planned stops that depends on the chosen shift.

### Changed

- Checkpoints no longer store the search space, which the optimizer loading one already has.

## 0.9.0 (2026-10-31)

The genetic optimizer, retuned: adaptive mutation, no repeat evaluations and blend crossover. Against 0.8.2, its geometric-mean error on the benchmark's 36 problems falls to ×0.58, ×0.23, ×0.16, ×0.17 and ×0.012 at 10, 20, 50, 100 and 1,000 evaluations per dimension. Results of existing genetic optimizer runs change. Benchmark results are in [benchmarks/README.md](benchmarks/README.md).

### Changed

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
