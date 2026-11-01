# hone
Python-based optimization engine.

Documentation: https://greatemuwar.github.io/hone/

## Install

```
pip install hone-python
```

It needs Python 3.12 or later. The package is named `hone-python` because `hone` is taken on PyPI; in code it is still `import hone`.

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

Each generation waits for its slowest evaluation before the next one starts, so workers beyond the population size sit idle, and so do workers that finish early while one slow evaluation runs on. When evaluation times vary, set `min_finished` in the config to let the next generation start once that share of the current one has finished:

```python
config = CMAESConfig(max_evaluations=2000, min_finished=0.75)
```

The slowest quarter of each generation then finishes in the background while the workers move on, and each late result joins the generation under way when it arrives: the genetic optimizer can breed from it, Bayesian optimization adds it to its model, and every optimizer counts it toward the best result. At the end, `optimize()` waits for the last of them, so the whole budget is used; stopping early by callback or threshold doesn't wait. Because results then depend on which evaluations finish first, runs with `min_finished` below 1 can't be repeated exactly.

Measured with 8 workers on evaluations of uneven length (one in ten taking five times as long), `min_finished=0.75` gets the same number of evaluations done in 23% to 43% less time. Per evaluation it does a little worse for CMA-ES and Bayesian optimization, whose updates prefer complete generations, and better for the genetic optimizer; but in the same time, with the evaluations it saves, its median error was ×0.76 for the genetic optimizer, ×0.25 for CMA-ES and ×0.12 for Bayesian optimization. Lower values, such as 0.5, save little more time and cost more per evaluation.

Workers beyond the population size still sit idle. CMA-ES allows for this: unless you set `population_size`, it uses at least twice as many candidates per generation as there are workers. Bayesian optimization proposes one candidate per worker at a time. Their results, unlike the other optimizers', therefore depend on the number of workers as well as the seed. For the genetic optimizer and random search, keep `population_size` at least two or three times `workers`.

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

### Following a run

`callback` is called after each generation with a `Progress`: the generation's statistics, the evaluations so far, the best value and parameters so far, and the seconds elapsed. `print_progress` prints one line per generation:

```python
from hone import print_progress

result = optimizer.optimize(run_simulation, workers=8, callback=print_progress)
# generation 0: 50 evaluations, best 12.4, 61.3 s
```

Return `True` from a callback to stop the run, for a stopping rule of your own:

```python
def good_enough_or_overnight(progress):
    good_enough = progress.best_fitness is not None and progress.best_fitness < 0.5
    return good_enough or progress.elapsed > 8 * 3600
```

Pressing Ctrl+C stops the run too. Either way, `optimize()` stops the workers and returns the best result so far, with `stopped` set.

### Resuming a run

A run of days shouldn't be lost to a crash, a reboot or a killed job. Give `optimize()` a checkpoint file:

```python
result = optimizer.optimize(run_simulation, workers=8, checkpoint="design-study.checkpoint")
```

After each generation the run's whole state is saved to the file. Run the same script again and the run carries on from the last generation saved, instead of starting over; the result is the same as if it had never stopped. Once the run has finished, running it again returns its result at once. The optimizer must be set up as before, with the same config, search space and initial points; a checkpoint from a different setup is refused.

With ask and tell, call `optimizer.save_checkpoint(path)` whenever you like, and `optimizer.load_checkpoint(path)` on a new optimizer set up the same way. Trials that were handed out but not yet told are handed out again by the next `ask()`, and can still be told by their ids.

A checkpoint is a pickle, which can run code when it is loaded, so only load checkpoints you saved yourself.

### Failed evaluations

Simulations crash, time out or fail to converge. Report such a run as failed rather than inventing a penalty value:

```python
optimizer.tell(trial, failed=True)
```

Inside `optimize()`, an objective function reports a failure by raising `hone.EvaluationFailed` or returning NaN or infinity; any other exception stops the run, so bugs in the function aren't hidden. Failed evaluations count toward `evaluations` and `failures` in the result, rank below every successful one, and are never used to breed new candidates.

### Noisy objectives

When the same parameters can give different values, as with a stochastic simulation, a measurement on a physical rig or a model trained with random initialization, set `noise=True`:

```python
config = GeneticOptimizerConfig(max_evaluations=2000, noise=True, final_evaluations=100)
```

Otherwise the optimizer believes every value it sees, and its best result is likely to be a point that got lucky. With `noise=True`:

- Every evaluation of the same point is averaged, and a point's fitness is that average. The genetic optimizer often breeds a point again, which re-evaluates it for free, and ranks its parents by their latest averages.
- Bayesian optimization measures improvement against the best value its model predicts at any evaluated point, rather than against the luckiest single value.
- `final_evaluations` keeps that many evaluations back for the end, to evaluate the most promising points again (up to 5 of them, each at least twice) and return the one with the best average. Its `best_fitness` is then an average too, and so a fairer estimate of what the result will achieve. It's off by default, because on small budgets the evaluations are worth more to the search.

`python -m hone.benchmarks --noisy` measures this: it adds noise to every evaluation and scores each run by the true, noise-free value of the point it returns.

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
| `RangeParameter` | A number between `minimum` and `maximum`; `integer=True` restricts it to whole numbers, and `log=True` searches it on a log scale | `RangeParameter("rate", 0, 1)` |
| `ChoiceParameter` | One of a list of `choices`, or a list of `choose_n` distinct choices | `ChoiceParameter("kernel", ["linear", "rbf"])` |
| `FixedParameter` | A single value that never changes | `FixedParameter("mode", "fast")` |

Use `log=True` for settings that span orders of magnitude, such as learning rates, regularization strengths or tree counts. Values are then spread evenly in their logarithm, so 0.001 to 0.01 is searched as thoroughly as 100 to 1000, and every optimizer searches the range that way. A log range needs a positive minimum, and works with `integer=True`:

```python
RangeParameter("learning_rate", 1e-5, 1e-1, log=True)
RangeParameter("n_estimators", 10, 1000, integer=True, log=True)
```

Every parameter the objective function requires must be in the search space. The optimizer checks this before it starts.

### Constraints

Limits across parameters are written with the parameters themselves and passed to `SearchSpace`. Every optimizer then only asks for points that satisfy every constraint, so the objective is never evaluated outside them:

```python
from hone import RangeParameter, SearchSpace

width = RangeParameter("width", 1, 10)
height = RangeParameter("height", 1, 10)
search_space = SearchSpace([width, height], constraints=[width * height <= 40, width >= 2 * height])
```

Range parameters and numbers combine with `+`, `-`, `*`, `/`, `**` and `abs()`, and `<=` or `>=` makes the result a constraint; `<` and `>` mean the same. A point where a constraint can't be computed, such as one that divides by zero, counts as breaking it. There are no equality constraints: an equality leaves nothing to sample, so remove a variable instead and compute it inside the objective. Write a range as two constraints, since Python reads `0 <= width - height <= 5` as two comparisons joined by `and`, and hone raises an error rather than keep one half.

For anything an expression can't say, such as a square root or a choice parameter, use a function, which takes parameters by name and is satisfied when it returns zero or less:

```python
from hone import Constraint

def stress(width, height, material):
    return bending_stress(width, height) - ALLOWED[material]

search_space = SearchSpace([width, height, material], constraints=[Constraint.function(stress)])
```

Constraints should be quick to check: hone checks them for every candidate it considers, which can be many times the number of evaluations. Expression constraints are saved with a result; a function constraint saves only its name, so after `Result.load` pass the function again with `rebuild_optimizer(functions={"stress": stress})`.

Starting points must satisfy the constraints. If the constraints leave nothing to search, or a region too small to find in 100,000 random draws, the optimizer raises `InfeasibleSearchSpace` when it's created, before any evaluation. Each optimizer keeps to constraints in its own way: random search and Bayesian optimization only consider points that satisfy them, CMA-ES draws a candidate again while it breaks one, and the genetic optimizer moves such a child back toward its fitter parent until it is feasible, which tends to land it near the boundary, where constrained optima usually are. On the benchmark's constrained problems the choice between optimizers follows the same pattern as without constraints; see [benchmarks/README.md](benchmarks/README.md#constrained-problems). [examples/pressure_vessel.py](examples/pressure_vessel.py) designs a pressure vessel of least cost under three constraints and finds the best published design. [examples/maintenance_schedule.py](examples/maintenance_schedule.py) chooses a maintenance schedule for a production line, scored by a noisy simulation of a year, with a constraint on technician hours and a function constraint that depends on the chosen shift; it halves the expected yearly cost of a sensible schedule chosen by hand.

### Conditional parameters

Some settings only exist when another takes certain values: a kernel's settings only for the model that has kernels, a third layer's size only with three layers or more. Write each condition as `test >> parameters`, read "then", and pass them to `SearchSpace`:

```python
from hone import ChoiceParameter, RangeParameter, SearchSpace

model = ChoiceParameter("model", ["svm", "forest"])
kernel = ChoiceParameter("kernel", ["linear", "rbf"])
C = RangeParameter("C", 1e-3, 1e3, log=True)
gamma = RangeParameter("gamma", 1e-5, 1, log=True)
trees = RangeParameter("trees", 10, 500, integer=True)

search_space = SearchSpace(
    [model, kernel, C, gamma, trees],
    conditions=[
        model["svm"] >> [kernel, C],
        kernel["rbf"] >> gamma,  # only with an svm too, since kernel is
        model["forest"] >> trees,
    ],
)

def train(model, kernel=None, C=None, gamma=None, trees=None):
    ...
```

The left of `>>` is a test: a choice parameter indexed by the values that switch the parameters on, such as `model["svm"]` or `optimizer["sgd", "nesterov"]`, or a comparison of range parameters in brackets, such as `(layers >= 3) >> units_3`. The brackets are needed, since Python reads `layers >= 3 >> units_3` as `layers >= (3 >> units_3)`; hone raises an error saying so. The right is a parameter or a list of them, and each parameter can have one condition. Conditions nest: a parameter whose test uses a parameter that is switched off is off too.

A parameter that is switched off isn't part of the point. The objective isn't passed it, so give it a default, which hone checks before the run starts; results, progress and `best_params` leave it out; and a starting point must leave it out too. Two points that differ only in a parameter switched off are the same point, so they are averaged together with `noise=True` and the genetic optimizer doesn't count one as new.

A constraint applies only while every parameter it uses is switched on, and `test >> constraint` makes a constraint apply only while the test holds:

```python
depth = RangeParameter("depth", 2, 20, integer=True)
search_space = SearchSpace(
    [model, trees, depth],
    constraints=[model["forest"] >> (trees * depth <= 2000)],
    conditions=[model["forest"] >> [trees, depth]],
)
```

Every optimizer supports conditions; CMA-ES only conditions on range parameters, since it doesn't search choices. Bayesian optimization gains the most: a parameter switched off looks the same to its model whatever value it holds, so values that don't matter add no noise. On the benchmark's conditional problems its error is ×0.0004 that of searching every parameter all the time, at 10 evaluations per dimension. With conditions the genetic optimizer breeds each branch within its own niche by default (`niche_share` of 0.5, below), so a wide, easy branch doesn't crowd out a narrow, better one; see [benchmarks/README.md](benchmarks/README.md#conditional-problems). [examples/choose_a_model.py](examples/choose_a_model.py) chooses a classifier, among an SVM with three kernels, a random forest and logistic regression, together with its settings.

### Configuration

`GeneticOptimizerConfig` validates its values when created. It can also be loaded from a JSON or YAML file:

```python
config = GeneticOptimizerConfig.from_file("config.yaml")
```

Set `seed` to make a run repeatable. `max_evaluations` sets a budget: the run stops after that many evaluations, even partway through a generation, which makes runs of different optimizers directly comparable. Given a budget, the genetic optimizer chooses its own population size, about a twentieth of the budget between 16 and 50, and runs until the budget is spent, unless you set `population_size` or `generations` yourself. Without a budget they default to 100 and 25.

Parents are chosen by tournament: `tournament_size` (default 4) individuals are drawn and the fittest wins.

When a choice parameter's best option is found late, the genetic optimizer can settle on another one: once most of the population uses one option, a child with a different option inherits numbers tuned for the wrong one and loses. `niche_share=0.5` breeds half of each generation's children within niches of parents that share their choices, so every option keeps tuning its own numbers. On material-selection-10d with 1,000 evaluations per dimension it found the right material in 10 of 10 runs, against 6 of 10 without. It costs some progress on small budgets, so it's off by default, except in a search space with conditions: there each branch is a niche, and 0.5 is the default, since a branch with a better optimum is often harder to find.

Each value of a new individual mutates with probability `mutation_rate`: by default 1 over the number of variables, and at least 0.1, so in two variables half of them mutate. A range value moves by a random step whose typical size is a fraction of the width of its range. A choice switches to another option, and a fixed value never changes.

The step size adapts as the run goes (`adaptive_mutation`, on by default). Each individual carries its own step, starting at `mutation_scale` (default 0.1); a child takes the geometric mean of its parents' steps, varied at random, so individuals whose steps suit the problem's current scale breed more, and steps shrink as the population closes in. Together with the mutation rate, this cut the genetic optimizer's geometric-mean error on the benchmark by half or more at 20 to 1,000 evaluations per dimension (×0.34 to ×0.50), and by 8% at 10. With `noise=True` both stay as before (a rate of 0.1 and a fixed step), because on the noisy problems they helped at small budgets but cost more at large ones; set `adaptive_mutation=True` or a `mutation_rate` to choose otherwise.

A child that would repeat a point already asked for, such as a copy of a parent that no mutation touched, is mutated again, one value at a time, until it is new. About one child in four used to be such a copy, and evaluating it again only spent the budget; avoiding them cut the geometric-mean error on the benchmark by 10% to 18% at 10 to 100 evaluations per dimension, and by 59% at 1,000. In a small integer space, where the points near the best run out, some repeats remain. With `noise=True` repeats are kept, because each one refines the point's average.

A child's numbers come from blend crossover (`blend_crossover`, on by default): each range value is drawn from between its two parents' values, and up to `blend_alpha` (default 0.1) times their distance beyond either, rather than copied from one parent. Choices and fixed values are still copied, each from the first parent with probability `crossover_rate`. On the benchmark this cut the genetic optimizer's geometric-mean error to ×0.69, ×0.62, ×0.51, ×0.38 and ×0.08 at 10 to 1,000 evaluations per dimension, and from 50 evaluations per dimension up it now beats random search on every problem. It helps with noise too (×0.67 and ×0.84 at 10 and 40 evaluations per dimension on the noisy problems, unchanged at 200), so it stays on with `noise=True`.

### Results

`optimize()` and `result()` return a `Result`:

| Field | Meaning |
| --- | --- |
| `best_fitness`, `best_params` | The best individual found in the whole run |
| `generations` | One `GenerationStats` per generation: the evaluations used so far, its best fitness and parameters, worst fitness, mean, median and standard deviation |
| `evaluations`, `failures` | How many evaluations were made, and how many of them failed |
| `reached_threshold` | Whether the run stopped early because it reached `threshold` |
| `stopped` | Whether the run was stopped before it finished, by its callback or Ctrl+C |
| `run` | How the result was produced: the optimizer, its full config including the seed, the search space, any initial points, the objective's name, `workers` and `timeout`, the versions of hone, Python and numpy, and when the run started and how long it took |

A result can be saved to JSON and loaded back:

```python
result.save("run.json")
result = Result.load("run.json")
```

`rebuild_optimizer()` sets up a new optimizer exactly as the result's was, so a saved run can be repeated. Run it on the same objective and it returns the same result; for CMA-ES and Bayesian optimization, also use the same number of workers, `result.run.workers`. An optimizer class of your own is rebuilt with `rebuild_optimizer(MyOptimizer)`.

```python
optimizer = Result.load("run.json").rebuild_optimizer()
```

Two results are equal when they found the same things: the same best point, generations and evaluations. Their `run` records aren't compared, since a repeated run starts at a different time.

### Optimizers

| Optimizer | Config | Approach |
| --- | --- | --- |
| `BayesianOptimizer` | `BayesianConfig` | Models the objective with a Gaussian process and evaluates where the model expects the most improvement. The first choice when evaluations are expensive and the budget is a few hundred evaluations or fewer. Supports range and choice parameters |
| `CMAES` | `CMAESConfig` | Learns the scale of each variable and how variables interact. The first choice for continuous and integer problems with larger budgets; doesn't support choice parameters |
| `GeneticOptimizer` | `GeneticOptimizerConfig` | Selection, crossover and mutation. Handles every kind of parameter, so it's the choice for problems with choice parameters and larger budgets |
| `RandomSearch` | `OptimizerConfig` | Evaluates random points; useful as a baseline |

```python
from hone import CMAES, CMAESConfig

result = CMAES(search_space, CMAESConfig(max_evaluations=5000, seed=0)).optimize(run_simulation)
```

CMA-ES chooses its own population size from the number of variables and workers, and by default restarts with a larger population when a run converges or stalls. Giving it a budget with `max_evaluations` is usually the only setting it needs.

Bayesian optimization spends a fraction of a second choosing each candidate, rising to a second or so after a few hundred evaluations. That is time well spent when an evaluation is a simulation or a training run that takes minutes, and wasted on a function that takes microseconds. Its first `2 × (variables + 1)` candidates are spread over the search space; after that it proposes one candidate at a time, or one per worker with `optimize(workers=n)`, and `population_size` sets that batch size directly. [benchmarks/results.md](benchmarks/results.md) compares the optimizers.

`choose_optimizer` picks one for you, by what the benchmark shows works best for the budget and the kinds of parameter:

```python
from hone import choose_optimizer

optimizer = choose_optimizer(search_space, max_evaluations=150, seed=0)  # BayesianOptimizer
result = optimizer.optimize(run_simulation)
```

| Budget | Parameters | Chosen |
| --- | --- | --- |
| 200 evaluations or fewer | Any, unless a choice parameter has `choose_n` | `BayesianOptimizer` |
| More than 200 | Any choice or integer parameter | `GeneticOptimizer` |
| More than 200 | Continuous numbers only | `CMAES` |

Other settings, such as `seed`, `maximize`, `noise` or `threshold`, are passed on to the chosen optimizer's config. On the benchmark's 180 problem and budget pairs, its pick had a median error 1.37 times the best optimizer's, on geometric mean, and was the best on 146; always using CMA-ES gives 7.9 times the best, and it can't run the 15 problems with choice parameters. Its biggest misses are plain Rastrigin with 1,000 evaluations per dimension and plain ellipsoid with 50 to 100, where the genetic optimizer wins because the variables don't interact, which a rule can't know in advance; on the rotated versions CMA-ES wins.

All four take the same arguments and support `optimize()`, `ask()` and `tell()`. A new optimizer subclasses `Optimizer` and implements `_next_population`, which returns each generation's individuals; the base class handles handing out candidates, recording results and failures, tracking the best result, stopping and the `Result`.

### Several objectives

When a design is judged on more than one thing, such as accuracy and cost, or mass and stiffness, a weighted sum forces you to price one against the other before seeing what the trade-off looks like, and some trade-offs no weighting can reach. Name each objective and its direction instead, and have the objective function return a dict of them. The run returns the **front**: every design found for which no other is at least as good on every objective and better on one. You choose from it afterwards.

```python
from hone import NSGA2, NSGA2Config

config = NSGA2Config(objectives={"accuracy": "maximize", "cost": "minimize"}, max_evaluations=2000, seed=0)

def train(**params):
    ...
    return {"accuracy": accuracy, "cost": cost}

result = NSGA2(search_space, config).optimize(train)
choice = result.front.best("accuracy", where={"cost": 20})  # the most accurate design costing at most 20
print(choice.params, choice.values)
```

`optimize()` returns a `ParetoResult`, whose `front` holds `ParetoPoint`s, each with its `params` and `values`. To choose from it:

| Method | Returns |
| --- | --- |
| `front.best(objective, where={...})` | The best point on one objective, with the others at most (or, for those maximized, at least) the limits given |
| `front.closest(weights)` | The point nearest the ideal, by augmented Chebyshev distance on objectives scaled from the front's best value of each to its worst. Unlike a weighted sum it can pick any point, including one in a dent in the front. Weights default to equal |
| `front.ideal`, `front.nadir` | The best and the worst value of each objective on the front |
| `front.hypervolume(reference)` | The region the front dominates, up to a reference point: one number that grows as the front gets closer and wider |

| Optimizer | Config | Approach |
| --- | --- | --- |
| `MultiObjectiveBayesian` | `MultiObjectiveBayesianConfig` | Each candidate draws random weights for the objectives and goes where a Gaussian process expects the most improvement in their weighted combination. The first choice for 200 evaluations or fewer. `method="parego"`, the default, models the combination; `method="models"` models each objective, which did better on three objectives at 20 evaluations per dimension and costs a few times more computation per candidate. Supports range and choice parameters |
| `NSGA2` | `NSGA2Config` | A genetic algorithm that ranks the population by fronts and spreads it along them. The choice for larger budgets; breeds as the genetic optimizer does, so it handles every kind of parameter |
| `MultiObjectiveRandomSearch` | `MultiObjectiveConfig` | Evaluates random points; useful as a baseline |

`choose_pareto_optimizer(search_space, objectives, max_evaluations)` picks `MultiObjectiveBayesian` up to 200 evaluations, unless a choice parameter has `choose_n`, and `NSGA2` otherwise, as the benchmark suggests; other settings are passed on to the chosen optimizer's config. On it, Bayesian optimization's median front beats NSGA2's on 12 of the 13 problem and budget pairs both run on and don't tie.

Everything else works as with one objective: constraints, conditions, workers and time limits, failed evaluations, checkpoints, `ask()` and `tell(trial, {...})`, starting points, `save()`, `load()` and `rebuild_optimizer()`. With `noise=True`, each point on the front is an average, with its number of evaluations and the `standard_error` of each objective. `print_progress` prints the size of the front and its hypervolume. A dict with missing or extra keys, or a value that isn't a number, raises `TypeError`; `maximize`, `threshold` and `final_evaluations` don't apply.

[examples/cantilever_beam.py](examples/cantilever_beam.py) designs a beam in steel, aluminium or carbon fibre that is light, cheap and stiff, three objectives, under a stress limit that depends on the material, and chooses the stiffest design under a mass and a cost limit. Tuning for more than three objectives is out of scope for now: with many objectives nearly every design is on the front, and the search loses its pull.

### Benchmarks

`hone.benchmarks` runs optimizers on test problems with known optima and reports how close each one gets:

```
python -m hone.benchmarks                  # full suite: 36 problems, 10 seeds each, five budgets
python -m hone.benchmarks --quick          # 6 problems, 5 seeds, three budgets; takes seconds
python -m hone.benchmarks --budgets 10 20  # only the budgets you name, in evaluations per dimension
python -m hone.benchmarks --multi          # several objectives: 7 problems, scored by hypervolume gap
```

The suite has five standard continuous functions (sphere, ellipsoid, Rosenbrock, Rastrigin and Ackley) in 2, 5 and 10 dimensions, rotated versions of four of them in which the variables interact, and integer, mixed and categorical problems: 36 in all. Each function is shifted so its optimum is away from the centre of the search space. Every optimizer gets exactly the same number of evaluations. Each problem runs at five budgets, 10, 20, 50, 100 and 1,000 evaluations per dimension, because the best optimizer for a slow simulation that allows a few dozen runs is often not the best one for a cheap function evaluated thousands of times. Latest results are in [benchmarks/results.md](benchmarks/results.md).

To compare your own optimizer, pass a factory that builds it for a problem, a budget and a seed. The budget is the run's total number of evaluations, so the factory can choose settings to suit it:

```python
from hone.benchmarks import run_benchmark, standard_problems, summarize, to_markdown

runs = run_benchmark(standard_problems(), optimizers={"mine": make_my_optimizer})
print(to_markdown(summarize(runs)))
```

#### Machine learning tuning

`python -m hone.benchmarks --ml` (or `make benchmark-ml`) tunes real models, each scored by 5-fold cross-validation on a dataset bundled with scikit-learn, which these problems need (`pip install scikit-learn`):

| Problem | Model and data | Searched |
| --- | --- | --- |
| svm-breast-cancer | Support vector classifier, 569 breast tumour samples | C and gamma |
| kernel-ridge-diabetes | Kernel ridge regression of disease progression, 442 patients | alpha, gamma and the kernel |
| random-forest-wine | 20-tree random forest, 178 wines in 3 classes | depth, leaf size, features per split and split criterion |

The best possible score is unknown, so error is measured from the best score a long reference search found. They run at 10, 20 and 50 evaluations per dimension, the budgets model tuning usually has. Results are in [benchmarks/ml-results.md](benchmarks/ml-results.md).

[examples/tune_iris.py](examples/tune_iris.py) shows the same kind of tuning in a few lines. C and gamma span orders of magnitude, so they are log-scale ranges:

```python
def accuracy(C, gamma, kernel):
    model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel=kernel))
    return cross_val_score(model, X, y, cv=folds).mean()

search_space = [
    RangeParameter("C", 1e-3, 1e3, log=True),
    RangeParameter("gamma", 1e-4, 10, log=True),
    ChoiceParameter("kernel", ["rbf", "sigmoid"]),
]
result = BayesianOptimizer(search_space, BayesianConfig(max_evaluations=30, maximize=True)).optimize(accuracy)
```

## Development

Requires Python 3.12 or later and [Poetry](https://python-poetry.org/) 2.

| Command | Does |
| --- | --- |
| `make test` | Install the dependencies and run the tests with coverage |
| `make lint` | Run ruff, then mypy to check types |
| `make docs` | Serve the documentation site at http://127.0.0.1:8000, rebuilding as files change (see [Docs](#docs)) |
| `make docs-build` | Build the documentation site into `site/`, failing on any warning, as CI does |
| `make docker-test` | Build the Docker image and run the tests inside it (see [Running in Docker](#running-in-docker)) |
| `make benchmark` | Run the full benchmark suite and update `benchmarks/results.md` |
| `make benchmark-noisy` | Run the noisy problems and update `benchmarks/noisy-results.md` |
| `make benchmark-ml` | Run the machine learning tuning problems and update `benchmarks/ml-results.md` |
| `make benchmark-constrained` | Run the constrained problems and update `benchmarks/constrained-results.md` |
| `make benchmark-conditional` | Run the conditional problems and update `benchmarks/conditional-results.md` |
| `make benchmark-multi` | Run the multi-objective problems and update `benchmarks/multi-results.md` |

### Docs

The documentation site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) from `mkdocs.yml` and the pages in `docs/`. Most pages hold no text of their own: a line such as `<!-- include README.md#constraints level=2 -->` pulls in a section of this README, and others pull in `CHANGELOG.md`, the benchmark results and the examples, so each is written once. `docs/hooks.py` does the including and points links at the site's own pages. The API reference is generated from docstrings by mkdocstrings; a config field is documented by a docstring on the line after it. `docs/design/` holds design notes and isn't published.

On every push to `main`, CI builds the site and publishes it to the `gh-pages` branch with [mike](https://github.com/jimporter/mike), as the version `latest`, which GitHub Pages serves. For it to be served, the repository's Pages settings must publish from the `gh-pages` branch.

Only `latest` is kept for now. To keep the docs of each release once that's needed:

1. Add `extra: {version: {provider: mike}}` to `mkdocs.yml`, for the version menu.
2. Change the deploy step to publish each release as its own version, with `latest` as an alias: `mike deploy --update-aliases 0.13 latest`, run when a version is released.
3. The first time, run `mike delete latest` just before that deploy, in the same job, since a version and an alias can't share a name.

Links to `/latest/...` keep working throughout, now pointing at the newest release.

### Running in Docker

With [Docker](https://docs.docker.com/get-docker/) installed you can run hone without Python or Poetry on your machine. The image holds hone with its development dependencies, the tests, the examples and the saved benchmark summaries, on Python 3.14.

```sh
make docker-build                    # build the image, tagged hone
make docker-test                     # run the tests in it
make docker-shell                    # a shell in it, with hone installed
docker run --rm hone python examples/pressure_vessel.py
```

To run a script of your own, mount its folder and run it from there:

```sh
docker run --rm -v "$PWD:/work" -w /work hone python my_search.py
```

To run a benchmark and keep its results, mount the `benchmarks` folder so the files it writes land in your copy:

```sh
docker run --rm -v "$PWD/benchmarks:/src/benchmarks" hone python -m hone.benchmarks --quick --markdown benchmarks/quick-results.md
```

Rebuild the image after changing hone's code; changing only the code reuses the cached dependency layer, so it takes seconds.

To release, raise `version` in `pyproject.toml` and add a section for it to `CHANGELOG.md`. Once the change is on `main` and its checks pass, CI tags the version, publishes a GitHub release with that section as its notes and the built package attached, and uploads the same package to PyPI as `hone-python`.

`pbt/tests/` holds property tests, which check hone against what its documentation promises over search spaces, configs and objectives that Hypothesis generates; set `PBT_EXAMPLES` (default 25) for a deeper search. `pbt/SUMMARY.md` lists every property and where it is promised.

CI also runs the quick benchmark and fails if any optimizer's median error, at any budget, grows more than tenfold against `benchmarks/quick-baseline.json`. After an intended change in results, update the baseline with `python -m hone.benchmarks --quick --save benchmarks/quick-baseline.json`.

## Security

Please report vulnerabilities privately, not as issues: see [SECURITY.md](SECURITY.md) for how, and for what counts.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
