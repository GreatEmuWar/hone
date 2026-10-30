# hone
Python-based optimization engine.

## Use

```python
from hone import GeneticOptimizer, GeneticOptimizerConfig, RangeParameter


def objective_function(x, y):
    return x**2 + y**2


# Each parameter of the objective function is declared with the values it may take.
x = RangeParameter("x", -5, 5)
y = RangeParameter("y", -5, 5)

optimizer = GeneticOptimizer(
    objective_function=objective_function,
    search_space=[x, y],
    config=GeneticOptimizerConfig(threshold=0.01, seed=0),
)

result = optimizer.optimize()
print(result.best_fitness, result.best_params)
```

The optimizer minimizes by default; set `maximize=True` in the config to maximize instead.

### Parameters

| Class | Values | Example |
| --- | --- | --- |
| `RangeParameter` | A number between `minimum` and `maximum`; `integer=True` restricts it to whole numbers | `RangeParameter("rate", 0, 1)` |
| `ChoiceParameter` | One of a list of `choices`, or a list of `choose_n` distinct choices | `ChoiceParameter("kernel", ["linear", "rbf"])` |
| `FixedParameter` | A single value that never changes | `FixedParameter("mode", "fast")` |

Every parameter the objective function requires must be in the search space. The optimizer checks this before it starts.

### Configuration

`GeneticOptimizerConfig` validates its values when created. It can also be loaded from a JSON or YAML file:

```python
config = GeneticOptimizerConfig.from_file("config.yaml")
```

Set `seed` to make a run repeatable.

Each value of a new individual mutates with probability `mutation_rate` (default 0.1). A range value moves by a random step whose typical size is `mutation_scale` (default 0.1) times the width of its range. A choice switches to another option, and a fixed value never changes.

### Results

`optimize()` returns a `Result`:

| Field | Meaning |
| --- | --- |
| `best_fitness`, `best_params` | The best individual found in the whole run |
| `generations` | One `GenerationStats` per generation: its best fitness and parameters, worst fitness, mean, median and standard deviation |
| `evaluations` | How many times the objective function was called |
| `reached_threshold` | Whether the run stopped early because it reached `threshold` |

A result can be saved to JSON and loaded back:

```python
result.save("run.json")
result = Result.load("run.json")
```

### Optimizers

| Optimizer | Config | Approach |
| --- | --- | --- |
| `GeneticOptimizer` | `GeneticOptimizerConfig` | Selection, crossover and mutation, keeping the best of each generation |
| `RandomSearch` | `OptimizerConfig` | Evaluates random points; useful as a baseline |

Both take the same arguments and return the same `Result`. A new optimizer subclasses `Optimizer` and implements `_next_population`, which returns each generation's individuals; the base class handles evaluation, tracking the best result, stopping and the `Result`.
