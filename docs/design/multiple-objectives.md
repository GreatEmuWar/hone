# Multiple objectives: design

Status: signed off. Part 1, the base-class refactor (#37), and part 2, the objectives, fronts, hypervolume and the random baseline, are in, and part 3 adds NSGA2, MultiObjectiveBayesian, `choose_pareto_optimizer` and the benchmark. The benchmark kept ParEGO as the default, with one model per objective as `method="models"`, which did better on three objectives (see benchmarks/README.md). Part 4, the README section, a worked example and 0.12.0, follows.

## What users get

Several objectives optimized together, such as accuracy and cost, and the set of best trade-offs, the Pareto front, rather than one answer to a weighted score. A design is on the front when no other design found is at least as good on every objective and better on one. The choice between the designs on it is made after seeing what each costs.

```python
from hone import NSGA2, NSGA2Config

config = NSGA2Config(objectives={"accuracy": "maximize", "cost": "minimize"}, max_evaluations=2000, seed=0)

def train(**params):
    ...
    return {"accuracy": accuracy, "cost": cost}

front = NSGA2(search_space, config).optimize(train).front
choice = front.best("accuracy", where={"cost": 20})  # the most accurate design costing at most 20
```

A weighted sum, the way to do this until now, fixes the weights before seeing the trade-off, and can't reach the parts of a front that bend inward, whatever the weights.

## Decisions (signed off)

### 1. Named objectives, and a dict from the function

`objectives` maps each objective's name to `"minimize"` or `"maximize"`. The function returns a dict with exactly those keys, and `tell(trial, {...})` takes one. Names appear in results and can't be swapped the way a tuple's order can. A dict with missing or extra keys, or a value that isn't a number, raises `TypeError`, as a non-number does today: it is a bug in the objective, not a failed evaluation. `EvaluationFailed`, NaN and infinity still mark a failed evaluation, which ranks behind every other.

### 2. Separate optimizer classes

`NSGA2` and `MultiObjectiveBayesian`, with a random-search baseline, return a `ParetoResult`. A mode on the existing classes would make `optimize()` return `Result | ParetoResult`, a type no overload can narrow from a config value, which would break every typed caller although nothing else changed. Separate classes keep `Result`, its type and every single-objective optimizer exactly as they are. They reuse the genetic optimizer's variation, repair, conditions and niches by composition.

### 3. The front comes from an archive

NSGA-II isn't monotone: once the front holds more points than the population, crowding drops non-dominated points, so a trade-off found early can be lost. The result's front is every non-dominated point evaluated, kept in an archive; hone's budgets make an unbounded archive affordable. Objective values equal within a small relative tolerance count as equal, so floating-point noise doesn't add spurious points.

### 4. Choosing from the front

- `front.best(objective, where={...})`: the best value of one objective with limits on the others, the ε-constraint method.
- `front.closest(weights)`: the point closest to the ideal by augmented Chebyshev distance, on objectives scaled to the front's own range from its ideal to its nadir. It can pick any point on the front, including those in a dent that a weighted sum can't reach, and the scaling is stated rather than implied.

Each point carries its number of evaluations, and with noise its standard error: a front of averages favours lucky points, which averaging alone doesn't undo.

### 5. How each optimizer handles it

- **NSGA2**: parents ranked by non-domination, then crowding distance, with the surviving front as elites. Ties break by evaluation id, so seeded runs repeat exactly and the result doesn't depend on the number of workers.
- **MultiObjectiveBayesian**: ParEGO, a random Chebyshev weighting of the normalized objectives per candidate fitted by one Gaussian process, against one Gaussian process per objective with the weighting applied to their predictions; the benchmark decides. A Chebyshev scalarization has kinks a single smooth model fits poorly, which is why the second is worth trying.
- **CMA-ES** isn't extended: MO-CMA-ES is a separate algorithm.
- `choose_optimizer` gains a multi-objective counterpart: Bayesian up to 200 evaluations, NSGA2 above.

### 6. Measured by hypervolume

The hypervolume of a front is the region of objective space it dominates, up to a reference point; a better front always scores higher. Each benchmark problem has a fixed normalization and reference point from its known front, so scores compare across runs and versions. The suite: ZDT1 to ZDT4 and DTLZ2, shifted so their optimal designs aren't on the bounds, which clipping and blend crossover would otherwise favour, and rotated so the variables interact; the welded beam's cost against its deflection, under its constraints; and an ML problem, accuracy against model size. The baseline is the best of several weighted-sum runs sharing the same total budget. Results show median attainment surfaces, the usual picture of a distribution of fronts, not only medians.

### 7. Out of scope for the first version

`threshold`, `final_evaluations` (noise averaging stays), and tuning for more than three objectives, where domination loses its pull and crowding distance weakens.

## Plan

Four PRs, then 0.12.0:

1. Split the optimizer base class into a core shared by every optimizer, `OptimizerCore`, and the single-objective `Optimizer`. No result changes: 242 runs across every optimizer and feature give identical results, and a checkpoint saved by 0.11.0 resumes identically.
2. The objectives setting, dict values, `ParetoResult` and its methods, the archive, hypervolume and the random baseline.
3. `NSGA2` and `MultiObjectiveBayesian`, with the multi-objective benchmark suite.
4. A README section, a worked example, and the bump.
