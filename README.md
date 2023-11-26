# hone
Python-based optimization engine.

## use
```python

from hone.config import GeneticOptimizerConfig
from hone.constraint import Constraint
from hone.objective_function import ObjectiveFunction
from hone.optimizers import GeneticOptimizer


def objective_function(x, y):
    return x**2 + y**2

seed_parameters = { 'x': 1, 'y': 1 }
constraints = [
    Constraint(param='x', minimum=-5, maximum=5),
    Constraint(param='y', minimum=-5, maximum=5)
]
# Default configuration for this example
config = GeneticOptimizerConfig()

optimizer = GeneticOptimizer(
    objective_function=objective_function,
    seed_parameters=seed_parameters,
    constraints=constraints,
    config=config
)

result = optimizer.optimize() 

```

