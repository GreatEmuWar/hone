# Conditional parameters: design

Status: signed off. Part 1, the notation, the search space and random search, is in; the other optimizers and the benchmark follow.

## What users get

Parameters that exist only while others take certain values, written with `>>`, read "then":

```python
from hone import ChoiceParameter, RangeParameter, SearchSpace

model = ChoiceParameter("model", ["svm", "forest"])
kernel = ChoiceParameter("kernel", ["linear", "rbf"])
C = RangeParameter("C", 0.01, 100, log=True)
gamma = RangeParameter("gamma", 1e-4, 1, log=True)
trees = RangeParameter("trees", 10, 500, integer=True)

space = SearchSpace(
    [model, kernel, C, gamma, trees],
    conditions=[
        model["svm"] >> [kernel, C],
        kernel["rbf"] >> gamma,  # nested: gamma needs kernel, which needs an svm
        model["forest"] >> trees,
    ],
)
```

The objective is called with only the parameters switched on at each point, so it gives the others defaults: `def train(model, kernel=None, C=None, gamma=None, trees=None)`.

The workaround until now was to keep every parameter on and have the objective ignore the irrelevant ones. That spends evaluations on points that differ only in an ignored value, and gives a model, such as Bayesian optimization's, noise to fit. The benchmark compares each optimizer with that workaround.

## Decisions (signed off)

### 1. `test >> parameters`

The left side is a test:

- `model["svm"]`: a choice parameter indexed by the values that switch the parameters on; `optimizer["sgd", "nesterov"]` for any of several.
- `(layers >= 3)`: a comparison of range parameters, as in a constraint, in brackets.

The right side is a parameter or a list of them. `>>` returns a `Condition`, a pydantic model that goes in `SearchSpace(conditions=...)` and saves with it, like a constraint. Parameters are frozen, so `>>` changes none of them, and a parameter can be defined before or after the one it depends on.

- `==` can't make a test: pydantic compares models with it, so overriding it would break the equality of search spaces and results. Indexing and `>>` are free on parameters.
- Python binds `>>` tighter than comparisons, so `layers >= 3 >> units_3` means `layers >= (3 >> units_3)`. That reaches `units_3.__rrshift__(3)`, which raises a `TypeError` saying to put the comparison in brackets.
- A parameter can have only one condition. Two, such as `optimizer["sgd"] >> momentum` and `optimizer["nesterov"] >> momentum`, leave open whether both must hold or either; the error suggests `optimizer["sgd", "nesterov"] >> momentum`.
- Chains such as `model["svm"] >> kernel["rbf"] >> gamma` aren't supported: whether the middle term is a parameter or a test is unclear. Separate lines nest by themselves, since a parameter whose parent is switched off is off too.
- Choice parameters with `choose_n` can't be tested in this version.
- A loop, such as two ranges each switching the other on, is refused when the search space is built.

### 2. The objective isn't passed parameters that are switched off

Trials, `best_params`, each generation's `best_params` and progress all leave them out, and `SearchSpace.active(params)` gives the point for any values. An objective without a default (or `**kwargs`) for a conditional parameter is refused before the run starts, with the fix. This matches `Constraint.function`, which also passes only what a function takes. Passing `None` instead would make every objective handle it.

### 3. A point is its active values

Two sets of values that differ only where parameters are switched off are the same point. The key behind noise averaging, the final re-evaluations and the genetic optimizer's no-repeat check uses only the active values.

### 4. Each optimizer

Every individual an optimizer breeds holds a value for every parameter, including those switched off at its point; the base class leaves them out of what it hands out and reports. That lets each optimizer keep its own representation:

| Optimizer | Conditional parameters |
| --- | --- |
| Random search | Draws every parameter; the ones switched off are left out of the trial |
| Genetic optimizer | Keeps a value for every gene, switched off or not, so a child that switches a branch back on inherits its old settings. Mutation, and the no-repeat re-mutation, only pick genes switched on, so none is wasted. Niches already group parents by their choice values, so a branch breeds within itself |
| Bayesian optimization | One Gaussian process over every column, with switched-off columns set to a fixed value, so the values they happen to hold add no noise. Heavier approaches, such as a model per branch, only if the benchmark calls for them |
| CMA-ES | Refuses choice parameters already, so it only meets conditions on ranges. It keeps those dimensions and ignores them while they are switched off |

`choose_optimizer` already sends spaces with choice parameters to Bayesian optimization at 200 evaluations or fewer and to the genetic optimizer above, and the benchmark checks that still holds.

### 5. Constraints on conditional parameters

A constraint applies only while every parameter it uses is switched on: `momentum * lr <= 0.1` is skipped when there is no momentum. A function constraint is skipped while a parameter it requires is switched off; it gets only the parameters switched on, so an optional one takes its default. With a constraint on the right, `>>` states a condition outright: `optimizer["sgd"] >> (momentum * lr <= 0.1)` applies only with sgd, and saves like any other constraint.

### 6. Initial points and `check`

A point gives values for the parameters switched on, and only those: a value for one switched off is an error naming the test that switches it on ("C isn't used at this point, since it's only used when model is 'svm': leave it out"). `best_params` from a previous run works as it is. An optimizer fills an initial point's switched-off parameters with random values, for its own use.

## Measuring it

A conditional suite joins the benchmark:

- Synthetic: pick one of several branches, each a standard function on its own dimensions with a different offset, so the best branch has to be found and then optimized; a nested version; and a layer count whose larger values switch on more dimensions.
- Machine learning: pick one of the three scikit-learn models the ML suite tunes and that model's settings, by cross-validation.

Each optimizer runs every problem twice: with conditions, and with the workaround of every parameter switched on. The comparison is the result.

## Plan

Three PRs, then 0.11.0:

1. The `>>` notation, `SearchSpace(conditions=...)`, `active` and `check`, conditional constraints, the point key, the objective's defaults, and random search. Other optimizers refuse conditions until part 2.
2. The genetic optimizer, Bayesian optimization and CMA-ES, with the conditional benchmark suite.
3. A README section, a worked example, and the bump.

## Out of scope for now

Tests of `choose_n` parameters, a parameter switched on by either of two different parents, chained `>>`, and per-branch models.
