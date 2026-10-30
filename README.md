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

optimizer = GeneticOptimizer(search_space=[x, y], config=GeneticOptimizerConfig(threshold=0.01, seed=0))
result = optimizer.optimize(objective_function)
print(result.best_fitness, result.best_params)
```

The optimizer minimizes by default; set `maximize=True` in the config to maximize instead.

### Slow evaluations

When each evaluation takes a while, run them in parallel. `workers` evaluates that many candidates at a time in background processes; the objective function must be defined at the top level of a module, and on macOS and Windows the calling script needs an `if __name__ == "__main__":` guard.

```python
result = optimizer.optimize(run_simulation, workers=8, timeout=600)
```

`timeout` limits each evaluation, in seconds. An evaluation that runs longer is stopped and counts as failed, as does one whose process crashes, so one hung simulation can't stall the run. A timeout also runs evaluations in a background process when `workers` is 1.

Each generation waits for its slowest evaluation before the next one starts, so workers beyond the population size sit idle. CMA-ES allows for this: unless you set `population_size`, it uses at least twice as many candidates per generation as there are workers. That means its results, unlike the other optimizers', depend on the number of workers as well as the seed. For the genetic optimizer and random search, keep `population_size` at least two or three times `workers`.

When evaluations run somewhere `optimize()` can't reach, such as jobs on a compute cluster or tests on a physical rig, ask the optimizer for candidates and tell it the results yourself, in any order:

```python
while not optimizer.done:
    for trial in optimizer.ask():
        submit_job(trial.id, trial.params)
    for job in wait_for_finished_jobs():
        optimizer.tell(job.trial_id, job.value)

result = optimizer.result()
```

`ask()` hands out the candidates of the current generation; `ask(4)` hands out at most four. It returns an empty list while the generation still waits for results, because the next generation is bred from them. `result()` can be called at any point for the best result so far.

### Failed evaluations

Simulations crash, time out or fail to converge. Report such a run as failed rather than inventing a penalty value:

```python
optimizer.tell(trial, failed=True)
```

Inside `optimize()`, an objective function reports a failure by raising `hone.EvaluationFailed` or returning NaN or infinity; any other exception stops the run, so bugs in the function aren't hidden. Failed evaluations count toward `evaluations` and `failures` in the result, rank below every successful one, and are never used to breed new candidates.

### Starting points

If you already have a good design, or the best parameters from an earlier run, pass it as a starting point so the search doesn't begin from scratch:

```python
optimizer = CMAES(search_space, CMAESConfig(max_evaluations=500), initial_points={"thickness": 2.5, "width": 40})

# or continue from an earlier result
optimizer = GeneticOptimizer(search_space, config, initial_points=[earlier.best_params])
```

`initial_points` takes one dict of parameter values or a list of them; fixed parameters may be left out, and every value is checked against the search space. The points are evaluated first and count toward the budget. The genetic optimizer and random search put them in their first generation, filling the rest with random candidates. CMA-ES evaluates them, then centres its search on the best one; lower `initial_step` to keep the search close to it.

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

Set `seed` to make a run repeatable. `max_evaluations` sets a budget: the run stops after that many evaluations, even partway through a generation, which makes runs of different optimizers directly comparable. Given a budget, the genetic optimizer chooses its own population size, about a twentieth of the budget between 16 and 50, and runs until the budget is spent, unless you set `population_size` or `generations` yourself. Without a budget they default to 100 and 25.

Each value of a new individual mutates with probability `mutation_rate` (default 0.1). A range value moves by a random step whose typical size is `mutation_scale` (default 0.1) times the width of its range. A choice switches to another option, and a fixed value never changes.

### Results

`optimize()` and `result()` return a `Result`:

| Field | Meaning |
| --- | --- |
| `best_fitness`, `best_params` | The best individual found in the whole run |
| `generations` | One `GenerationStats` per generation: the evaluations used so far, its best fitness and parameters, worst fitness, mean, median and standard deviation |
| `evaluations`, `failures` | How many evaluations were made, and how many of them failed |
| `reached_threshold` | Whether the run stopped early because it reached `threshold` |

A result can be saved to JSON and loaded back:

```python
result.save("run.json")
result = Result.load("run.json")
```

### Optimizers

| Optimizer | Config | Approach |
| --- | --- | --- |
| `CMAES` | `CMAESConfig` | Learns the scale of each variable and how variables interact. The first choice for continuous and integer problems; doesn't support choice parameters |
| `GeneticOptimizer` | `GeneticOptimizerConfig` | Selection, crossover and mutation. Handles every kind of parameter, so it's the choice for problems with choice parameters |
| `RandomSearch` | `OptimizerConfig` | Evaluates random points; useful as a baseline |

```python
from hone import CMAES, CMAESConfig

result = CMAES(search_space, CMAESConfig(max_evaluations=5000, seed=0)).optimize(run_simulation)
```

CMA-ES chooses its own population size from the number of variables and workers, and by default restarts with a larger population when a run converges or stalls. Giving it a budget with `max_evaluations` is usually the only setting it needs. [benchmarks/results.md](benchmarks/results.md) compares the three optimizers.

All three take the same arguments and support `optimize()`, `ask()` and `tell()`. A new optimizer subclasses `Optimizer` and implements `_next_population`, which returns each generation's individuals; the base class handles handing out candidates, recording results and failures, tracking the best result, stopping and the `Result`.

### Benchmarks

`hone.benchmarks` runs optimizers on test problems with known optima and reports how close each one gets:

```
python -m hone.benchmarks                  # full suite: 36 problems, 10 seeds each, five budgets
python -m hone.benchmarks --quick          # 6 problems, 5 seeds, three budgets; takes seconds
python -m hone.benchmarks --budgets 10 20  # only the budgets you name, in evaluations per dimension
```

The suite has five standard continuous functions (sphere, ellipsoid, Rosenbrock, Rastrigin and Ackley) in 2, 5 and 10 dimensions, rotated versions of four of them in which the variables interact, and integer, mixed and categorical problems: 36 in all. Each function is shifted so its optimum is away from the centre of the search space. Every optimizer gets exactly the same number of evaluations. Each problem runs at five budgets, 10, 20, 50, 100 and 1,000 evaluations per dimension, because the best optimizer for a slow simulation that allows a few dozen runs is often not the best one for a cheap function evaluated thousands of times. Latest results are in [benchmarks/results.md](benchmarks/results.md).

To compare your own optimizer, pass a factory that builds it for a problem, a budget and a seed. The budget is the run's total number of evaluations, so the factory can choose settings to suit it:

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

CI also runs the quick benchmark and fails if any optimizer's median error, at any budget, grows more than tenfold against `benchmarks/quick-baseline.json`. After an intended change in results, update the baseline with `python -m hone.benchmarks --quick --save benchmarks/quick-baseline.json`.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
