# Bug Report: hone.optimizers.base With noise, the threshold is checked against a stale average, so runs stop early and report `reached_threshold` when the best average misses the threshold

**Target**: `hone.optimizers.base.Optimizer._finish_generation`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2026-10-31

## Summary

With `noise=True`, a point's fitness is the average of every evaluation of it. `Optimizer._best` holds the best individual, and its `fitness` is the average *as it was when that individual was told*. Later evaluations of the same point create new individuals and raise the point's average, but they leave `_best.fitness` unchanged. `_finish_generation` checks `threshold` against this stale `_best.fitness`. As a result, one lucky evaluation can stop the run early with `reached_threshold=True`, even though the result's `best_fitness` (the current best average) has not reached the threshold.

## Property-Based Test

```python
@run_settings
@given(problems(), st.floats(min_value=0.0, max_value=0.3))
def test_noisy_threshold_consistent(problem, threshold):
    kind, space, seed, budget = problem
    objective = Recorder(space, noise=random.Random(seed))
    result = build(kind, space, seed=seed, max_evaluations=budget, noise=True, threshold=threshold).optimize(objective)
    if result.reached_threshold:
        assert result.best_fitness <= threshold
```

(`pbt/tests/test_pbt_base.py`. `Recorder` is a deterministic objective over the search space, plus Gaussian noise with a standard deviation of 0.05.)

**Failing input**: `problem=('genetic', SearchSpace([ChoiceParameter('_', [0])]), 114, 10), threshold=0.0`. The test reported `best_fitness=0.00099 > threshold=0.0` with `reached_threshold=True`. The same failure occurs with `SearchSpace([RangeParameter('n', 0, 10, integer=True)])`, seed 10 and a budget of 200: that run stopped after 16 of 200 evaluations.

## Reproducing the Bug

```python
from hone import OptimizerConfig, RandomSearch, RangeParameter

config = OptimizerConfig(population_size=4, noise=True, threshold=-0.5, seed=0)
optimizer = RandomSearch([RangeParameter("n", 0, 1, integer=True)], config)
trials = optimizer.ask()
print([t.params for t in trials])  # [{'n': 1}, {'n': 1}, {'n': 1}, {'n': 0}]

values = iter([-1.0, 1.0, 1.0, 1.0])
for trial in trials:
    optimizer.tell(trial, next(values))

result = optimizer.result()
print(result.reached_threshold, result.best_fitness, optimizer.done)
# True 0.3333333333333333 True
```

The point `n=1` was evaluated three times, giving -1, 1 and 1, so its average is 1/3. `n=0` gave 1. No point's average is at or below -0.5. Even so, the run stops after its first generation of a default 25 and reports `reached_threshold=True`, with `best_fitness=0.333`.

The same happens through `optimize()` on a realistic noisy problem:

```python
import random
from hone import GeneticOptimizer, GeneticOptimizerConfig, RangeParameter

noise = random.Random(3)
def objective(n):
    return (n - 3) ** 2 + noise.gauss(0, 1)

config = GeneticOptimizerConfig(max_evaluations=200, noise=True, threshold=-0.5, seed=3)
result = GeneticOptimizer([RangeParameter("n", 0, 10, integer=True)], config).optimize(objective)
print(result.reached_threshold, result.best_fitness, result.evaluations)
# True -0.37093089833408505 30
```

The run spends only 30 of its 200 evaluations, then stops. It claims to have reached the threshold of -0.5, but its reported best average is -0.37.

## Why This Is A Bug

- The config defines `threshold` as "stop once the best fitness reaches this". The README defines `reached_threshold` as "Whether the run stopped early because it reached `threshold`".
- With noise, the README says "a point's fitness is that average", and the result's `best_fitness` "is then an average too, and so a fairer estimate of what the result will achieve". The best fitness the run reports therefore never reached the threshold, yet the run stopped and said it had.
- The point of `noise=True` is to stop believing lucky single values: "Otherwise the optimizer believes every value it sees, and its best result is likely to be a point that got lucky." The stopping rule still believes a lucky value, using an average that later evaluations have superseded. Users with noisy objectives and a threshold lose most of their budget and get a result that misses their target. Nothing tells them, because `reached_threshold` says the target was met.

Runs without noise are unaffected: there, `_best.fitness` is never stale.

## Fix

Check the threshold against the best value the result reports. Without noise, `_best_point()` returns the same value as before; with noise, it returns the current best average.

```diff
--- a/hone/optimizers/base.py
+++ b/hone/optimizers/base.py
@@ -495,7 +495,8 @@ class Optimizer(ABC):
     def _finish_generation(self, final: bool = False) -> None:
         assert self._population is not None
         self._history.append(self._summarize(self._generation, self._evaluations, self._population))
-        if self._best is not None and self._reached(_fitness(self._best)):
+        best = self._best_point()
+        if best is not None and self._reached(best[0]):
             self._reached_threshold = True
             self._finish()
         elif final or self._generation + 1 >= self.config.generations or self._budget_left() == 0:
```

With this patch, both reproductions above run on as they should: the first prints `False 0.333… False`, and the second finds no failing seed in 200. All 492 existing tests in `tests/` still pass.
