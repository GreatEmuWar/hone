"""Choosing an optimizer for a problem, by what the benchmark shows works best."""
from typing import Any

from hone.config import BayesianConfig, CMAESConfig, GeneticOptimizerConfig
from hone.parameters import ChoiceParameter, Parameter, SearchSpace

from .base import Optimizer
from .bayesian import BayesianOptimizer
from .cmaes import CMAES
from .genetic import GeneticOptimizer

# Up to this many evaluations, Bayesian optimization had the lowest median error on most
# benchmark problems; past it, each of its candidates costs too long to choose.
BAYESIAN_UP_TO = 200


def choose_optimizer(
    search_space: SearchSpace | list[Parameter],
    max_evaluations: int,
    *,
    initial_points: dict[str, Any] | list[dict[str, Any]] | None = None,
    **config: Any,
) -> Optimizer:
    """The optimizer the benchmark suggests for this search space and budget, set up to use it.

    - Up to 200 evaluations: Bayesian optimization, unless a choice parameter has choose_n.
    - More, with choice parameters: the genetic optimizer, since CMA-ES searches numbers only.
    - More, with numbers only: CMA-ES.

    Other settings, such as seed, maximize, noise or threshold, are passed on to the chosen
    optimizer's config. The optimizer's class and config show what was chosen.
    """
    if not isinstance(search_space, SearchSpace):
        search_space = SearchSpace(search_space)
    choices = [p for p in search_space.parameters if isinstance(p, ChoiceParameter)]
    settings = dict(config, max_evaluations=max_evaluations)
    if max_evaluations <= BAYESIAN_UP_TO and not any(p.choose_n is not None for p in choices):
        return BayesianOptimizer(search_space, BayesianConfig(**settings), initial_points=initial_points)
    if choices:
        return GeneticOptimizer(search_space, GeneticOptimizerConfig(**settings), initial_points=initial_points)
    return CMAES(search_space, CMAESConfig(**settings), initial_points=initial_points)
