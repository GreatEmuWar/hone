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
