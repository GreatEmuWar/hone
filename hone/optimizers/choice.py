"""Choosing an optimizer for a problem, by what the benchmark shows works best."""
from typing import TYPE_CHECKING, Any

from hone.config import BayesianConfig, CMAESConfig, GeneticOptimizerConfig
from hone.parameters import ChoiceParameter, Parameter, RangeParameter, SearchSpace

from .base import Optimizer
from .bayesian import BayesianOptimizer
from .cmaes import CMAES
from .genetic import GeneticOptimizer

if TYPE_CHECKING:
    from .multi import MultiObjectiveOptimizer

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
    - More, with integer parameters: the genetic optimizer, which did better on the benchmark's
      integer and mixed problems.
    - More, with continuous numbers only: CMA-ES.

    Other settings, such as seed, maximize, noise or threshold, are passed on to the chosen
    optimizer's config. The optimizer's class and config show what was chosen.
    """
    if not isinstance(search_space, SearchSpace):
        search_space = SearchSpace(search_space)
    choices = [p for p in search_space.parameters if isinstance(p, ChoiceParameter)]
    settings = dict(config, max_evaluations=max_evaluations)
    if max_evaluations <= BAYESIAN_UP_TO and not any(p.choose_n is not None for p in choices):
        return BayesianOptimizer(search_space, BayesianConfig(**settings), initial_points=initial_points)
    integers = [p for p in search_space.parameters if isinstance(p, RangeParameter) and p.integer and p.maximum > p.minimum]
    if choices or integers:
        return GeneticOptimizer(search_space, GeneticOptimizerConfig(**settings), initial_points=initial_points)
    return CMAES(search_space, CMAESConfig(**settings), initial_points=initial_points)


def choose_pareto_optimizer(
    search_space: SearchSpace | list[Parameter],
    objectives: dict[str, Any],
    max_evaluations: int,
    *,
    initial_points: dict[str, Any] | list[dict[str, Any]] | None = None,
    **config: Any,
) -> "MultiObjectiveOptimizer":
    """The multi-objective optimizer the benchmark suggests for this search space and budget, set up to use it.

    - Up to 200 evaluations: multi-objective Bayesian optimization, unless a choice parameter has choose_n.
    - More: NSGA2.

    objectives names each objective and its direction, as in MultiObjectiveConfig. Other
    settings are passed on to the chosen optimizer's config.
    """
    from hone.config import MultiObjectiveBayesianConfig, NSGA2Config

    from .genetic import NSGA2
    from .multi_bayesian import MultiObjectiveBayesian

    if not isinstance(search_space, SearchSpace):
        search_space = SearchSpace(search_space)
    settings = dict(config, objectives=objectives, max_evaluations=max_evaluations)
    choose_n = any(isinstance(p, ChoiceParameter) and p.choose_n is not None for p in search_space.parameters)
    if max_evaluations <= BAYESIAN_UP_TO and not choose_n:
        return MultiObjectiveBayesian(search_space, MultiObjectiveBayesianConfig(**settings), initial_points=initial_points)
    return NSGA2(search_space, NSGA2Config(**settings), initial_points=initial_points)
