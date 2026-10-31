# Constraints: design

Status: proposal, for sign-off before any code is written.

## What users get

Limits across parameters, written with the parameter objects themselves:

```python
from hone import RangeParameter, SearchSpace

width = RangeParameter("width", 1, 10)
height = RangeParameter("height", 1, 10)
space = SearchSpace([width, height], constraints=[width * height <= 40, width >= 2 * height])
```

Every optimizer then only ever asks for points that satisfy every constraint. An infeasible point is never evaluated, so a slow simulation never spends hours on a design that was ruled out before it started, and an objective never has to handle inputs outside its valid region.

## Decisions to sign off

### 1. Constraints are hard, and checked before evaluation

The alternative is a penalty: evaluate infeasible points and add a cost for the violation. That spends real evaluations on designs known to be invalid, and a penalty weight has to be tuned per problem. Constraints on the parameters are known before evaluation, so hone can enforce them exactly.

Constraints on the objective's *outputs* (the stress a simulation reports must stay below a limit) are different: they are only known after evaluating. They are out of scope here; today an objective can raise `EvaluationFailed` or add its own penalty. A later phase can let the objective return constraint values alongside its result, which Bayesian optimization can then model.

### 2. Expressions: arithmetic on range parameters, compared with `<=` or `>=`

- Range parameters, numbers and other expressions combine with `+`, `-`, `*`, `/`, `**`, unary minus and `abs()`. Integer and log ranges work; an expression uses the parameter's actual value.
- `<=` and `>=` turn an expression into a constraint. `<` and `>` mean the same: for numbers searched continuously the difference has no effect, and pretending otherwise invites rounding bugs.
- `==` is not supported. A continuous equality leaves a feasible region of zero volume, which no sampler can hit; the fix is to remove a variable, such as `depth = total - width` computed inside the objective. Overloading `==` would also break parameter equality, which pydantic and the checkpoint code rely on.
- `0 <= x + y <= 10` is a Python chained comparison that silently drops half of it. A constraint raises a `TypeError` with the fix ("write two constraints") if Python asks for its truth value, so the mistake can't pass unnoticed.
- Choice parameters can't appear in expressions, since their values aren't numbers. Constraints that involve choices use a function (below).

Expressions are pydantic models, so a search space with constraints still saves to JSON in a run record, compares equal for checkpoints, and rebuilds with `rebuild_optimizer()`.

### 3. A function for everything else

```python
from hone import Constraint

stress = Constraint.function(lambda width, height, material: max_stress(width, height, material) - LIMIT, name="stress")
```

The function takes parameters by name, like an objective, and the point is feasible when it returns zero or less. It covers square roots, lookups, choice parameters and anything else an expression can't say. The cost: a function can't be saved, so a run record stores its name, and `rebuild_optimizer()` and checkpoints need the same function passed in again.

### 4. Each optimizer keeps its own way of making candidates, and redraws infeasible ones

| Optimizer | Infeasible candidate |
| --- | --- |
| Random search | Drawn again |
| Genetic optimizer | Repaired: moved toward its fitter parent, which is feasible, by halving the gap ten times and keeping the last point inside, so it lands near the boundary. On the constrained benchmark this beat breeding again from new parents at every budget (×0.84 to ×0.44) |
| CMA-ES | Drawn again from the same distribution, up to 100 times, then replaced by a random feasible point. Ranking infeasible candidates below every feasible one without evaluating them was also tried: on the constrained benchmark it did worse at every budget (×1.17 to ×1.59) |
| Bayesian optimization | Its candidate pool keeps only feasible points, and a refined point that leaves the region is dropped |

Initial points must be feasible, or the optimizer raises `ValueError` when created, naming the constraint they break.

If a region is so small that 100,000 draws in a row find nothing feasible, the optimizer raises an error saying so, rather than looping. (1,000 was too few: the tension spring below is 0.8% feasible, so a streak of 1,000 misses turns up within a few runs.) At the start of a run hone draws feasible points from the whole space; if it can't find any, the error comes before a single evaluation, which also catches contradictory constraints. Samplers made for very thin regions, such as hit-and-run, are out of scope until a real problem needs one.

### 5. Measured like everything else

A constrained suite joins the benchmark, with known or best-known optima:

- Sphere and ellipsoid with a linear constraint that cuts off the unconstrained optimum, so the answer lies on the boundary, where it usually does in engineering.
- Rosenbrock inside a disk, the classic test whose optimum sits on the disk's edge.
- Three standard engineering problems: the pressure vessel (mixed integer), the tension spring and the welded beam. They use function constraints, since their formulas include square roots.

Each optimizer reports error at the usual budgets. Feasibility is guaranteed by construction, and the property tests check it: every point any optimizer asks for, on generated search spaces with generated constraints, satisfies them.

## Plan

Three PRs, then 0.10.0:

1. Expressions, `Constraint.function`, `SearchSpace(constraints=...)`, saving and checking them, feasible initial points, and random search. Property tests for the expression arithmetic.
2. The genetic optimizer, CMA-ES and Bayesian optimization, with the constrained benchmark suite, and the CMA-ES variant chosen by its results.
3. Documentation: a README section and a worked engineering example, such as the pressure vessel.

## Out of scope for now

Output constraints, equality constraints, soft constraints with tolerances, constraints between choice parameters as expressions, and thin-region samplers.
