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
| `generations` | One `GenerationStats` per generation: the evaluations used so far, its best fitness and parameters, worst fitness, mean, median and standard deviation |
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

### Benchmarks

`hone.benchmarks` runs optimizers on test problems with known optima and reports how close each one gets:

```
python -m hone.benchmarks            # full suite: 24 problems, 10 seeds each
python -m hone.benchmarks --quick    # 5 problems, 5 seeds; takes seconds
```

The suite has five standard continuous functions (sphere, ellipsoid, Rosenbrock, Rastrigin and Ackley) in 2, 5 and 10 dimensions, plus integer, mixed and categorical problems. Each function is shifted so its optimum is away from the centre of the search space. Every optimizer gets the same budget of evaluations. Latest results are in [benchmarks/results.md](benchmarks/results.md).

To compare your own optimizer, pass a factory that builds it for a problem, a budget and a seed:

```python
from hone.benchmarks import run_benchmark, standard_problems, summarize, to_markdown

runs = run_benchmark(standard_problems(), optimizers={"mine": make_my_optimizer})
print(to_markdown(summarize(runs)))
```

## Development

Requires Python 3.12 or later and [Poetry](https://python-poetry.org/) 2.

| Command | Does |
| --- | --- |
| `make test` | Install the dependencies and run the tests with coverage |
| `make lint` | Run ruff |
| `make docker-test` | Build the Docker image and run the tests inside it |
| `make benchmark` | Run the full benchmark suite and update `benchmarks/results.md` |

CI also runs the quick benchmark and fails if any optimizer's median error grows more than tenfold against `benchmarks/quick-baseline.json`. After an intended change in results, update the baseline with `python -m hone.benchmarks --quick --save benchmarks/quick-baseline.json`.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
