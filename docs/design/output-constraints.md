# Output constraints: design

Status: implemented, pending review.

## What users get

Limits on what an evaluation reports, written the same way as limits on the parameters and passed in the same place:

```python
from hone import BayesianOptimizer, Output, RangeParameter, SearchSpace

width, height = RangeParameter("width", 1, 10), RangeParameter("height", 1, 10)
stress = Output("stress")

space = SearchSpace(
    [width, height],
    constraints=[
        width * height <= 40,      # on the parameters: checked before evaluating, as before
        stress <= 250,             # on an output: checked once the evaluation reports it
        stress <= 50 * width,      # outputs and parameters can mix
    ],
)

def simulate(width, height):
    ...
    return {"objective": mass, "stress": max_stress}

result = BayesianOptimizer(space).optimize(simulate)
result.feasible       # whether best_params satisfies every constraint on the outputs
result.best_outputs   # {"stress": ...} at best_params
```

A parameter constraint keeps a slow simulation from spending hours on a design that was ruled out before it started. Many limits can't be known that early: the peak stress, the temperature, the latency, whatever the simulation computes. Until now an objective could only raise `EvaluationFailed` or add a penalty with a weight that has to be tuned for each problem. An output constraint is the third way: the optimizer is told how far each design breaks each limit and uses that.

## Decisions

### 1. One way to write a limit: `Output(name)` in `constraints=`

`Output("stress")` is a term like a range parameter, so it gets the same arithmetic, `abs()`, `<=`, `>=`, the `TypeError` for chained comparisons, `>>` for conditional limits, and pydantic saving. A limit that turns out to be computable from the parameters moves before evaluation by swapping one term for a formula, and nothing else changes.

A config dict, the alternative, would copy the way objectives are declared. It can't express a limit that mixes outputs and parameters or that applies only to some branches, and it would be a second notation for the same idea.

`Constraint.function(f, outputs=["stress"])` covers anything an expression can't say. The function is passed the parameters it names and the outputs listed in `outputs`. An output name can't also be a parameter name, so a name always means one thing.

### 2. What `constraints=` means now

A search space sorts its constraints by whether they use an output:

- `space.parameter_constraints`: checked before evaluation, exactly as before. `feasible()`, `broken()`, `sample()`, the initial-point check and `InfeasibleSearchSpace` use only these.
- `space.output_constraints`: checked once the evaluation reports its outputs. `space.outputs` is the set of names the objective must return.

`space.constraints` is the full list as written, which is what is saved and compared for checkpoints.

An output can't appear in a test, on the left of `>>`. A condition's test decides which parameters the objective is passed, which can't depend on what it returns. A conditional constraint's test could in principle, but its gate would switch on and off with a noisy output, and a model of the limit would have a step in it. `material["steel"] >> (stress <= 400)` is fine, since the test is on a parameter.

### 3. The objective returns a dict when there are output constraints

With output constraints, a single-objective function returns `{"objective": value, **outputs}`. A multi-objective function returns its objectives and the outputs in one dict. A name can be both an objective and an output, so `Output("cost") <= 20` limits an objective directly. `tell(trial, {...})` takes the same dict.

The dict must hold exactly those keys, with a number for each. Missing or extra keys, or a value that isn't a number, raise `TypeError`, since that is a bug in the objective. NaN or infinity anywhere marks the evaluation as failed, as it does for objectives today. `EvaluationFailed` works as before.

Without output constraints nothing changes: a single objective still returns a number.

### 4. Ranking: feasibility first, with no penalty weight

Every evaluation that succeeds is ranked by Deb's feasibility rule (Deb, 2000):

1. A feasible point beats an infeasible one.
2. Between feasible points, the better objective wins.
3. Between infeasible points, the smaller total violation wins.

A failed evaluation ranks below all of them.

The total violation is the sum, over the output constraints that apply, of each one's violation relative to the size of its limit. For `stress <= 250` that is `(stress - 250) / 250`. This makes limits in different units comparable: 10% over a stress limit counts the same as 10% over a deflection limit. A limit of zero, which includes every function constraint, has no size to divide by, so its violation counts as it is. A function constraint should return a relative margin, such as `stress / LIMIT - 1`, when it shares a problem with other limits.

There's no penalty weight to tune. A point that breaks a limit is never mistaken for a good one however small its objective, and among points that break limits the search is steered back toward the feasible region.

### 5. How each optimizer uses it

| Optimizer | Output constraints |
| --- | --- |
| Random search | Ranked by the rule above, which only decides the best point |
| Genetic optimizer | Ranked by the rule: elites, tournaments, niches and the parent a repaired child moves toward use it. Roulette selection gives a point that breaks a limit weight zero while any point satisfies them all, as the worst feasible point already has; when none does, weights fall with the violation |
| CMA-ES | Ranked by the rule, which decides the recombination weights. The restart test for a flat run looks only at feasible bests |
| Bayesian optimization | One Gaussian process per output constraint, of its margin. Expected improvement on the best feasible value is multiplied by the probability that every constraint holds (Gardner et al., 2014; Gelbart et al., 2014). Until a feasible point is found, it maximizes that probability alone |
| NSGA2 | Constrained domination (Deb et al., 2002): feasible points by fronts and crowding, then infeasible ones by violation |
| Multi-objective Bayesian | The same probability of feasibility multiplies the improvement in the weighted objectives, measured from the feasible points |

The models in Bayesian optimization learn where the infeasible region is from the evaluations that broke a limit. The other optimizers only learn which designs ranked worse.

A Gaussian process for a constraint is fitted to its relative margin, through the bilog transform `sign(m) log(1 + |m|)` (Eriksson and Poloczek, 2021). The margin is `left - right` for `left <= right`, the other way round for `>=`, and the function's value for a function constraint. The relative margin divides it by the limit's size, as the violation does. Bilog keeps the sign, so the limit stays at zero, and compresses the far tails: stresses near the edges of a design space run to many times their limit, and a stationary Gaussian process fitted to those raw values smooths over the boundary, which is the part that matters. On the welded beam example, modelling raw margins left one seed of four at 40% above the best design. With bilog all four came within 1.4%, with a quarter fewer evaluations breaking a limit.

A margin that couldn't be computed, such as from a division by zero, is modelled as the largest margin seen. A conditional constraint is fitted only to the points where it applies, and counts as satisfied wherever its test doesn't hold. The constraints' models are treated as independent, so the probability that all hold is the product of each one's.

### 6. What the result says

- `Result.best_params` is the best feasible point. If no evaluation was feasible, it is the point with the smallest violation, and `Result.feasible` is `False`.
- `Result.best_outputs` holds the outputs at that point, and `Result.infeasible` counts the evaluations that broke an output constraint.
- `GenerationStats` counts each generation's infeasible evaluations. Its fitness statistics cover only the feasible ones, so they don't mix in values from designs that were ruled out.
- A threshold is reached only by a feasible point.
- On a front, `ParetoPoint.outputs` holds each point's outputs, and the front contains only feasible points. While no point is feasible the front is empty and `ParetoResult.feasible` is `False`.

### 7. Not supported yet: noise

With `noise=True`, a point's outputs would have to be averaged with its objective, and whether it is feasible would become a probability that the final re-evaluation would have to settle. Without that, an optimizer would keep a design that passed its limit once by luck. Until it's designed, an optimizer refuses output constraints combined with noise with `UnsupportedSearchSpace`.

`sensitivity()` accepts output constraints. They don't change where it samples, so it measures the objective's value from the returned dict and ignores the outputs. Parameter constraints are still refused.

### 8. Measured like everything else

The constrained suite gains output-constraint versions of the pressure vessel, the tension spring and the welded beam. They are the same problems with their limits reported by the objective instead of written on the parameters. Each optimizer reports error at the usual budgets on both versions, which shows how much is lost by not knowing a limit before evaluating.

## Out of scope for now

Noise with output constraints, equality constraints, soft tolerances, tests on outputs, and constraint values that come from a different, cheaper evaluation than the objective.
