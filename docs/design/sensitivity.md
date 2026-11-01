# Sensitivity analysis: design

Status: in progress. Morris screening, Sobol indices and the report are built (part 1). Still to come in part 2: `report.reduced(space)`, a README section, a page on the documentation site and an example.

## What users get

Which parameters matter, measured before optimizing or instead of it:

```python
from hone import sensitivity

report = sensitivity(run_simulation, space, max_evaluations=500, workers=8, timeout=600)
print(report)
```

which prints, with illustrative numbers:

```
Sobol indices: 448 evaluations, 0 failed; 95% intervals
parameter   total        interval   first        interval
thickness   0.71   [0.62, 0.80]    0.55   [0.44, 0.66]
material    0.30   [0.24, 0.37]    0.19   [0.11, 0.27]
fillet      0.02   [0.01, 0.03]    0.01  [-0.02, 0.04]
```

An engineer with twelve parameters and a slow simulator learns which four drive the result, fixes the rest, and optimizes in a third of the dimensions. A study can also stop here: the answer the engineer needed may be "the fillet radius doesn't matter".

## Decisions (signed off)

### 1. One function, one report

`sensitivity(objective, space, max_evaluations=500)` returns a `SensitivityReport`. It holds each parameter's `ParameterSensitivity`, most important first; `report.ranking` gives the names in that order, `report["thickness"]` one parameter, and `print(report)` a table. It also holds the best point evaluated, which `maximize=` decides, since the evaluations would otherwise be thrown away.

The function lives in `hone/sensitivity_analysis.py` and is exported as `hone.sensitivity`. The module isn't called `sensitivity.py`, so `hone.sensitivity` is only ever the function, for users and for the API reference.

### 2. The same evaluation machinery as `optimize()`

`workers`, `timeout`, `EvaluationFailed`, NaN and infinity all behave as in `optimize()`, through the same worker pool. A failed evaluation is left out: for Morris, an effect needs both of its points; for Sobol, a row needs all of its points. Any other exception stops the analysis, as it stops a run.

### 3. Morris on small budgets, Sobol when the budget allows

- **Morris screening** (Morris 1991): random one-at-a-time trajectories on a four-level grid, with a step of 2/3 of each range. Each trajectory costs `parameters + 1` evaluations, and as many trajectories as fit in `max_evaluations` are used, at least two. The report gives `mu_star`, the mean size of each parameter's effect, with its sign as `mu` and its spread as `sigma`. Written in hone; it is a few lines.
- **Sobol indices**: Saltelli's design, `n × (parameters + 2)` evaluations with `n` a power of 2, drawn from scipy's scrambled Sobol sequence as `scipy.stats.sobol_indices` draws it, and first and total indices by the estimator of Saltelli et al. (2010), as scipy computes them. hone computes the estimator itself rather than calling `sobol_indices`, since `sobol_indices` needs a complete power-of-two sample and failed rows have to be dropped. A test checks that the two agree on the same evaluations.
- Without `method=`, Sobol is used when at least 64 rows fit in the budget, and Morris otherwise. On the Ishigami function, 64 rows is where the 95% intervals start to hold the true total index reliably: in 98% of cases over 20 seeds, against 90% at 32 rows. Asked for explicitly, Sobol needs at least 16 rows and Morris two trajectories; below that, the error says how many evaluations are needed.

No new dependency: scipy is already one.

### 4. Every parameter type, on the optimizers' scales

Each parameter is a coordinate from 0 to 1, mapped as the optimizers map it: a range by `from_unit`, so a log range evenly in its logarithm and an integer range rounded; a choice parameter by picking one of its choices in equal slices; a `choose_n` parameter by one of its combinations, in equal slices. An effect is therefore the change across the parameter's whole range. Fixed parameters are passed to the objective but not measured.

A parameter a condition switches off isn't passed to the objective there, so it has no effect at that point, which its indices include. That is the question a user asks: how much does `gamma` matter, given it only applies to one kernel.

### 5. Constraints refused at first

Morris and Sobol assume each parameter can take any of its values whatever the others are, which constraints break. `sensitivity()` raises `UnsupportedSearchSpace` for a space with constraints, saying why.

### 6. Intervals by bootstrap

Each interval is a percentile bootstrap, over trajectories for Morris and over rows for Sobol, at `confidence=0.95` by default, with 1,000 resamples drawn from the same seed. The percentile method rather than scipy's default BCa, because BCa fails when a parameter's effects are all the same, such as one the objective ignores.

### 7. Checked against published values

The tests check the Ishigami function's indices (Ishigami and Homma 1990) and Sobol's G-function's (Saltelli and Sobol 1995) against their exact values, Morris's ranking of the G-function, and exact effects of linear and log-linear objectives. Property tests check that every point is valid and within the budget, that a parameter the objective ignores scores exactly 0, and that Morris's effects are the change across each range.

### 8. Later

- `report.reduced(space)`: the search space with the unimportant parameters fixed, by default at the best point evaluated, ready for `choose_optimizer`. Part 2.
- Indices from a finished run's evaluations through a surrogate, at no extra evaluations.
- Tolerances around the best point: how far each parameter can move before the objective gets worse by a given amount.
- Constraints, through a method that allows dependent parameters, such as Shapley effects.

## Plan

1. Morris, Sobol, the report and their tests (this part).
2. `report.reduced(space)`, the README section, a guide page and an API page on the site, an example, and a release, 0.13.0.
