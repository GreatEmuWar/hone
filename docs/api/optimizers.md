# Optimizers

Every optimizer takes a search space, a config and optional starting points, and supports `optimize()`, `ask()` and `tell()`. See [Choosing an optimizer](../guide/optimizers.md) for which to use.

::: hone.choose_optimizer

::: hone.choose_pareto_optimizer

::: hone.BayesianOptimizer

::: hone.CMAES

::: hone.GeneticOptimizer

::: hone.RandomSearch

::: hone.MultiObjectiveBayesian

::: hone.NSGA2

::: hone.MultiObjectiveRandomSearch

## Base classes

::: hone.Optimizer
    options:
      heading_level: 3

::: hone.MultiObjectiveOptimizer
    options:
      heading_level: 3

## Evaluations

::: hone.Trial
    options:
      heading_level: 3

::: hone.EvaluationFailed
    options:
      heading_level: 3

::: hone.UnsupportedSearchSpace
    options:
      heading_level: 3
