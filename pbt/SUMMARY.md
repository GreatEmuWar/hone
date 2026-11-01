# Property-based testing round: summary

This was an independent bug hunt on hone 0.8.0. It followed the procedure in [agentic-pbt's `hypo.md`](https://github.com/mmaaz-git/agentic-pbt/blob/main/hypo.md): read the target, propose only properties that the code, its docstrings, README.md or CHANGELOG.md claim, test them with Hypothesis, triage every failure, and report only reproduced bugs that violate a claimed property. Nothing under `hone/` or `tests/` was changed.

## Results

**1 bug report**, in `pbt/bug_reports/`:

| Report | Target | Severity | Type | Summary |
| --- | --- | --- | --- | --- |
| [bug_report_hone_optimizers_base_2026-10-31_00-50_x440.md](bug_reports/bug_report_hone_optimizers_base_2026-10-31_00-50_x440.md) | `hone.optimizers.base` | Medium | Logic | With `noise=True`, the threshold is checked against a stale average of the best point. A lucky single evaluation can stop the run early with `reached_threshold=True` while the reported `best_fitness`, the current average, misses the threshold. In one realistic example, the run spent 30 of 200 evaluations. The report includes a two-line fix, which passes all 492 existing tests. **Fixed** in `hone/optimizers/base.py`. |

**Tests**: 93 property tests in `pbt/tests/`. All pass with the default settings. `test_noisy_threshold_consistent` covers the bug above, now fixed, and keeps its failing example as a regression check.

`make test` runs them along with `tests/`, since `pbt/tests` is in `testpaths` and Hypothesis is a dev dependency. To run them alone:

```
poetry run pytest pbt/tests                    # about 30 seconds; 25 examples per property
PBT_EXAMPLES=300 poetry run pytest pbt/tests   # a deeper search
```

During the hunt, every property was also run at 150 to 300 examples. `pbt_strategies.py` generates any valid parameter or search space. `pbt_runs.py` builds any optimizer on a generated problem, with an objective that records every call.

## Findings not reported as bugs

These failures were triaged and turned out not to be bugs, or not bugs that can be shown to break a claim. They are listed so that a later round doesn't repeat the work.

- **Elitism could round up to the whole population.** Elites were `round(population_size * elitism_rate)`, so a valid `elitism_rate` below 1 could make every individual an elite: `GeneticOptimizerConfig(max_evaluations=200, elitism_rate=0.97)` gave 16 elites in a population of 16, and the run ended after 16 of its 200 evaluations. It wasn't reported, because an existing test seemed to expect it; on triage that test turned out to be exercising the base class's skipping of empty generations, not elitism, so this was **fixed** afterwards: at most `population_size - 1` individuals are elites, and the skipping test now uses an optimizer that repeats its population.
- **Ties in the best value.** With different points tied for the best value, which one `best_params` names depends on the order of `tell()` calls: the first one told is kept. That is reasonable, and `test_tell_order_does_not_matter` compares `best_params` only when the best is untied.
- **Test-strategy artefacts**, not bugs: ranges one float ulp wide, the last generation cut short by the budget, and a parent set that left out carried-over elites. All were fixed in the tests.

## Properties tested, by target

**Source** says where each claim comes from. All properties pass. The one marked ✗ failed until the bug report's fix was applied.

### `hone.parameters` (`test_pbt_parameters.py`)

| Property | Source |
| --- | --- |
| `RangeParameter.sample` lies in the range, is an `int` for integer ranges, and `check()` returns it unchanged | `RangeParameter` docstring; README Parameters table |
| `from_unit` clips to the range and rounds for integer ranges, for any unit value | `from_unit` docstring |
| `to_unit(from_unit(u)) ≈ u` and `from_unit(to_unit(v)) ≈ v`, exactly for integers | `from_unit`: "The inverse of to_unit" |
| `to_unit` and `from_unit` give the same values for arrays as for scalars | "Accepts a number or an array" |
| `to_unit` is 0 at the minimum and 1 at the maximum | `to_unit` docstring |
| `mutate` stays within the bounds and keeps integers whole | `mutate` docstring: "The result is clipped to the bounds" |
| `check` rejects out-of-range values, booleans and non-numbers | `check` code and docstring |
| The validator rejects reversed bounds and log ranges with a minimum of 0 or below | `_check_bounds`; README: "A log range needs a positive minimum" |
| Log integer ranges only ever give integers in range | README: "works with `integer=True`" |
| `ChoiceParameter.sample` gives a choice, or `choose_n` distinct choices | `ChoiceParameter` docstring |
| A single-choice `mutate` always switches to a different choice | `mutate`: "switches to a different one when there is one to switch to" |
| `choose_n` `check` rejects repeated values, and `mutate` stays valid | `check` code; docstring |
| `FixedParameter` never changes | "A parameter held at a single value"; README |
| `SearchSpace.check` returns parameters in order, fills in left-out fixed values, and is idempotent | `check` docstring |
| `SearchSpace.check` rejects unknown parameters and missing non-fixed ones | `check` docstring |
| Repeated parameter names are refused | `_check_unique_names` |
| `SearchSpace` accepts any sequence, and round-trips through JSON | CHANGELOG 0.7.0; `Result.run` stores one |
| `space[name]` finds parameters and raises `KeyError` otherwise | `__getitem__` |

### `hone.config` (`test_pbt_config.py`)

| Property | Source |
| --- | --- |
| With a budget and no `population_size`, the genetic population is clamp(budget // 20, 16, 50), at least the tournament size, with enough generations to spend the budget | `GeneticOptimizerConfig` docstring; README Configuration |
| Settings you give explicitly win; without a budget, the defaults are 100 and 25 | Docstring; README |
| `BayesianConfig` sets `generations` to the budget | `BayesianConfig` docstring |
| `tournament_size` can't exceed `population_size`, and `final_evaluations` must be below `max_evaluations` | Validators |
| `reserved_evaluations` is `final_evaluations` with noise, and 0 without | Property docstring |
| Configs are frozen, and unknown settings are refused | `ConfigDict(frozen=True, extra="forbid")` |
| `from_file` with JSON and YAML gives back the config, and a partial file gives the same defaults as the constructor | README: "It can also be loaded from a JSON or YAML file" |
| `model_validate(model_dump(mode="json"))` gives back every config type | Relied on by `rebuild_optimizer` |
| Bounds on `min_finished` and `elitism_rate` | Field constraints and comments |

### `hone.result` (`test_pbt_result.py`)

| Property | Source |
| --- | --- |
| `save` then `load` gives an equal result, for generated results and for real runs of every optimizer, run record included | README Results: "A result can be saved to JSON and loaded back" |
| Equality ignores `run` but compares what was found | `Result` docstring; CHANGELOG 0.7.1 |
| `rebuild_optimizer()` after `load` gives the same class, config, search space and initial points, and running it gives an equal result | `rebuild_optimizer` docstring; README |
| `rebuild_optimizer` without a run record raises `ValueError` | Docstring and code |
| Results are frozen | `ConfigDict(frozen=True)` |
| `print_progress` prints one line in the README's format, with "none yet" before any success | `print_progress` docstring; README Following a run |

### `hone.optimizers.base` (`test_pbt_base.py`)

Each property was run on every optimizer that supports the generated search space: genetic, random search, CMA-ES and Bayesian.

| Property | Source |
| --- | --- |
| Every candidate handed out is a valid, ordered point of the search space | README: "every value is checked against the search space" |
| `evaluations` equals the number of objective calls and stays within `max_evaluations`; `failures` equals the failed calls (raise, NaN and inf) | README Results and Failed evaluations |
| A small budget is spent exactly | README: "stops after that many evaluations, even partway through a generation"; "the whole budget is used" |
| `best_fitness` is the best value seen, and `best_params` gives it; failures rank below successes, with minimize and maximize | README: "The best individual found in the whole run" |
| Generation stats are consistent: sequential indices, non-decreasing evaluations, best ≤ median and mean ≤ worst, best_params gives best_fitness, and the last generation's count equals the run's | `GenerationStats`; README Results |
| `reached_threshold` is true exactly when a value reached the threshold; the best then meets it; the run stops at the end of that generation | Config: "stop once the best fitness reaches this"; README |
| `maximize=True` on −f gives the mirror image of minimizing f | README: "set `maximize=True` ... to maximize instead" |
| The same seed gives the same result, including from the same optimizer twice | README: "Set `seed` to make a run repeatable"; `optimize` docstring |
| Driving the optimizer with ask and tell gives the same result as `optimize()` | Class docstring: "Use it either way round" |
| With `min_finished=1`, the order of tells within a generation doesn't change the result | README: "tell it the results yourself, in any order" |
| ask/tell contract: `ask(n)` returns at most n; `ask()` returns empty while a generation waits and once the run is done; tell rejects repeats, unknown ids, booleans and a missing value | `ask` and `tell` docstrings |
| NaN, ±inf and `failed=True` count as failures and are never best | `tell` docstring |
| Initial points are evaluated first, count toward the budget, and are recorded in `run` | README Starting points |
| The callback sees every generation, with the right evaluations, a best that never gets worse and elapsed time that never goes back; returning True stops the run with `stopped`, and the generations match an unstopped run's | README Following a run |
| Ctrl+C returns the best so far with `stopped`, and raises if nothing has succeeded | README; CHANGELOG 0.7.1 |
| A run that crashes and resumes from its checkpoint gives the same result as an uninterrupted one, with and without noise and final evaluations; a finished checkpoint returns at once without evaluating | README Resuming a run |
| A mid-generation ask/tell checkpoint hands out the outstanding trials again with the same ids, and continuing gives the same result | README: "Trials that were handed out but not yet told are handed out again" |
| A checkpoint from a different setup is refused | `load_checkpoint` docstring |
| With `min_finished < 1` under ask/tell, late results count toward the best, the whole budget is used, and the run isn't done until they are told | README Slow evaluations; `done` docstring |
| The same with worker processes in `optimize()` | README Slow evaluations |
| With noise, `best_fitness` is the average of every evaluation of `best_params` | README Noisy objectives |
| `final_evaluations`: the last evaluations re-evaluate up to 5 earlier points, each at least twice, and the result is the finalist with the best average | README Noisy objectives |
| ✗ With noise, `reached_threshold` implies the reported best reached the threshold | Config `threshold`; README `reached_threshold` and noise. **Bug report above; fixed** |
| The objective's signature is checked before any evaluation, and a non-numeric return raises `TypeError` | README Parameters: "The optimizer checks this before it starts" |

### `hone.optimizers.genetic` (`test_pbt_genetic.py`)

| Property | Source |
| --- | --- |
| Any valid config with a budget spends the whole budget (except elitism that rounds to the whole population; see above) | README: "runs until the budget is spent" |
| Generation 0 is the full population; each later one evaluates `population_size − elites` | Docstring: elites kept unchanged, the rest are children |
| With uniform crossover and without mutation, every gene of a child comes from a successfully evaluated individual | Docstring: crossover and mutation; "Failed evaluations are never elites or parents" |
| A choice option whose evaluations always fail is never bred, without mutation | Docstring; README Failed evaluations: "never used to breed new candidates" |
| With elitism, each generation's best never gets worse | Elites carried over unchanged |
| Results don't depend on the number of workers | `optimize` docstring: "The genetic optimizer and random search return it however many workers are used" |

Roulette selection, niches, tournament settings and maximization are all covered by the generated configs.

### `hone.optimizers.cmaes` (`test_pbt_cmaes.py`)

| Property | Source |
| --- | --- |
| Choice parameters are refused with `UnsupportedSearchSpace` | Class docstring |
| The default population is 4 + ⌊3 ln n⌋ every generation when restarts are off | `CMAESConfig` comment |
| With restarts, the population only grows by doubling | Module docstring (IPOP) |
| The search starts at the centre of every range | Class docstring |
| With initial points, they are evaluated first, and the search then centres on the best one | Class docstring; README Starting points |
| Fixed parameters and single-value ranges pass through unchanged | Class docstring |
| A 1-D integer search reaches its optimum, because the minimum integer spread lets it move | Module docstring (CMA-ES with Margin) |
| With workers, the population is at least twice the workers | `CMAESConfig` comment; README |

### `hone.optimizers.bayesian` (`test_pbt_bayesian.py`)

| Property | Source |
| --- | --- |
| `choose_n` is refused with `UnsupportedSearchSpace` | Class docstring; CHANGELOG 0.6.0 |
| The first batch is `initial_samples` or 2 × (variables + 1), initial points included and first; later batches are `population_size` or 1 | README Optimizers; `BayesianConfig` |
| No point is proposed twice on a continuous space, within or across batches | Code comments; CHANGELOG 0.8.0 |
| Points still outstanding with `min_finished < 1` are not proposed again, and late results count | CHANGELOG 0.8.0: "so it doesn't propose them again" |
| Runs with failed evaluations complete, and the best is the best success | Docstring: failures modelled as the worst value |
| It finds a 1-D smooth optimum within 20 evaluations | README: first choice for small budgets |

### `hone.optimizers.choice` (`test_pbt_choice.py`)

| Property | Source |
| --- | --- |
| The chosen optimizer follows the README table (Bayesian up to 200 without `choose_n`; genetic with choice or integer parameters; CMA-ES otherwise), including at the 200/201 boundary | `choose_optimizer` docstring; README table; CHANGELOG 0.8.0 |
| `max_evaluations`, `seed`, `maximize`, `noise` and `threshold` are passed on to the config | Docstring: "Other settings ... are passed on" |
| Initial points are passed on, after checking | Signature; README |
| The chosen optimizer runs and spends its budget | Implied by the README example |

### `hone.parameters.constraints` (`test_pbt_constraints.py`)

| Property | Source |
| --- | --- |
| An expression of range parameters and numbers evaluates to what the same arithmetic gives on plain numbers | `constraints` module docstring: "ordinary arithmetic" |
| A constraint saved to JSON and loaded again is equal to the original and judges every point the same | docs/design/constraints.md: "Expressions are pydantic models, so a search space with constraints still saves to JSON" |
| A violation is never negative, and a point satisfies a constraint exactly when its violation is 0 | `Constraint.violation` docstring |
| Every optimizer only asks for points that satisfy every constraint, for generated linear constraints | `SearchSpace` docstring: "Every point an optimizer asks for satisfies every constraint" |

### `hone.parameters.conditions` (`test_pbt_conditions.py`)

Search spaces are generated with up to six parameters, each possibly switched on by a test of an earlier one (a choice's values, or a range compared with a number), and listed in any order.

| Property | Source |
| --- | --- |
| The active point of a sample passes `check` unchanged, `active` leaves it as it is, and it keeps parameter order | `SearchSpace.active` and `check` docstrings |
| A parameter is switched on exactly when it has no condition or its test holds at the point, so a parameter whose parent is switched off is off too | `conditions` module docstring; docs/design/conditions.md |
| `check` refuses a value for a parameter that is switched off, and a point missing one that is switched on | `SearchSpace.check` docstring |
| A search space with conditions saved to JSON and loaded again is equal to the original and switches the same parameters on | `conditions` module docstring: "a search space with conditions saves and loads like any other" |
| Every optimizer only hands out points without switched-off parameters, and its best point is one too | `Optimizer` docstring |

### `hone.pareto` (`test_pbt_pareto.py`)

Generated sets of 1 to 40 points with 2 or 3 objectives, and random-search runs with every combination of directions.

| Property | Source |
| --- | --- |
| No point on the front dominates another point on it | `hone.pareto` module docstring |
| Every point is on the front, tied with a point on it, or dominated by one | `non_dominated` docstring |
| The front doesn't change under an increasing rescaling of any objective | docs/design/multiple-objectives.md |
| Adding a point never lowers hypervolume, and adding a dominated point leaves it unchanged | `Front.hypervolume` docstring: "A better front always has a larger hypervolume" |
| Hypervolume depends only on the front | Definition of hypervolume |
| A run's front is the front of every point it evaluated, with maximized objectives the right way round, and its hypervolume never falls | `MultiObjectiveOptimizer` docstring |
