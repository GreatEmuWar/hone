"""choose_optimizer: picking an optimizer by budget and parameters, and how well the picks do on the benchmark."""
import json
import math
import statistics
from pathlib import Path

import pytest
from pydantic import ValidationError

from hone import CMAES, BayesianOptimizer, GeneticOptimizer, choose_optimizer, choose_pareto_optimizer
from hone.config import BayesianConfig, CMAESConfig, GeneticOptimizerConfig, MultiObjectiveBayesianConfig, NSGA2Config
from hone.benchmarks import conditional_problems, constrained_problems, ml, noisy_problems, standard_problems
from hone.optimizers.genetic import NSGA2
from hone.optimizers.multi_bayesian import MultiObjectiveBayesian
from hone.parameters import ChoiceParameter, Output, RangeParameter, SearchSpace

NUMBERS = [RangeParameter("x", -5, 5), RangeParameter("n", 0, 10, integer=True)]
WITH_CHOICE = NUMBERS + [ChoiceParameter("material", ["steel", "titanium"])]
BENCHMARKS = Path(__file__).parent.parent / "benchmarks"
NAMES = {BayesianOptimizer: "bayesian", CMAES: "cma-es", GeneticOptimizer: "genetic"}


@pytest.mark.parametrize(
    "space, budget, expected",
    [
        (NUMBERS, 50, BayesianOptimizer),
        (NUMBERS, 200, BayesianOptimizer),
        (NUMBERS, 201, GeneticOptimizer),
        (NUMBERS[:1], 201, CMAES),
        ([RangeParameter("x", -5, 5), RangeParameter("n", 3, 3, integer=True)], 201, CMAES),
        (WITH_CHOICE, 100, BayesianOptimizer),
        (WITH_CHOICE, 5000, GeneticOptimizer),
        (NUMBERS + [ChoiceParameter("tags", ["a", "b", "c"], choose_n=2)], 50, GeneticOptimizer),
        (NUMBERS + [ChoiceParameter("only", ["one"])], 5000, GeneticOptimizer),
    ],
)
def test_the_choice_follows_budget_and_parameters(space, budget, expected):
    assert type(choose_optimizer(space, budget)) is expected


LIMIT = [Output("total") <= 4]
CHOOSE_N = ChoiceParameter("tags", ["a", "b", "c"], choose_n=2)


@pytest.mark.parametrize(
    "parameters, budget, expected",
    [
        (NUMBERS[:1], 201, BayesianOptimizer),
        (NUMBERS[:1], 300, BayesianOptimizer),
        (NUMBERS[:1], 500, BayesianOptimizer),
        (NUMBERS[:1], 501, CMAES),
        (NUMBERS, 500, BayesianOptimizer),
        (NUMBERS, 501, GeneticOptimizer),
        (WITH_CHOICE, 300, BayesianOptimizer),
        (NUMBERS + [CHOOSE_N], 300, GeneticOptimizer),
    ],
)
def test_output_constraints_without_noise_keep_bayesian_optimization_to_500(parameters, budget, expected):
    assert type(choose_optimizer(SearchSpace(parameters, LIMIT), budget)) is expected


def test_with_noise_the_cutoff_stays_at_200():
    space = SearchSpace(NUMBERS[:1], LIMIT)
    assert type(choose_optimizer(space, 200, noise=True)) is BayesianOptimizer
    assert type(choose_optimizer(space, 201, noise=True)) is CMAES
    assert type(choose_optimizer(space, 201, noise=False)) is BayesianOptimizer


def old_choice(parameters, budget):
    """The rule before output constraints raised the cutoff: the class and config it gave."""
    choices = [p for p in parameters if isinstance(p, ChoiceParameter)]
    integers = [p for p in parameters if isinstance(p, RangeParameter) and p.integer and p.maximum > p.minimum]
    if budget <= 200 and not any(p.choose_n is not None for p in choices):
        return BayesianOptimizer, BayesianConfig(max_evaluations=budget, seed=0)
    if choices or integers:
        return GeneticOptimizer, GeneticOptimizerConfig(max_evaluations=budget, seed=0)
    return CMAES, CMAESConfig(max_evaluations=budget, seed=0)


# Dense around the cutoffs, sparse elsewhere: building every budget from 1 to 10000 would be slow.
BUDGETS = sorted({*range(1, 11), *range(180, 221), *range(280, 321), *range(480, 521), 700, 1000, 2000, 5000, 10000})
UNCHANGED_SPACES = [
    NUMBERS[:1],
    NUMBERS,
    NUMBERS + [ChoiceParameter("material", ["steel", "titanium"])],
    NUMBERS + [CHOOSE_N],
    NUMBERS + [ChoiceParameter("material", ["steel", "titanium"]), CHOOSE_N],
]


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("parameters", UNCHANGED_SPACES, ids=range(len(UNCHANGED_SPACES)))
def test_nothing_else_moves(parameters, budget):
    expected_class, expected_config = old_choice(parameters, budget)
    optimizer = choose_optimizer(SearchSpace(parameters), budget, seed=0)
    assert type(optimizer) is expected_class and optimizer.config.model_dump() == expected_config.model_dump()


@pytest.mark.parametrize("budget", BUDGETS)
def test_choose_n_with_output_constraints_follows_the_old_rule(budget):
    parameters = NUMBERS + [CHOOSE_N]
    expected_class, expected_config = old_choice(parameters, budget)
    optimizer = choose_optimizer(SearchSpace(parameters, LIMIT), budget, seed=0)
    assert type(optimizer) is expected_class and optimizer.config.model_dump() == expected_config.model_dump()


@pytest.mark.parametrize("budget", [1, 100, 200, 201, 300, 500, 501, 5000])
@pytest.mark.parametrize("limits", [[], LIMIT])
@pytest.mark.parametrize("extra", [[], [CHOOSE_N]])
def test_choose_pareto_optimizer_is_unchanged(extra, limits, budget):
    objectives = {"a": "minimize", "b": "maximize"}
    optimizer = choose_pareto_optimizer(SearchSpace(NUMBERS + extra, limits), objectives, budget, seed=0)
    if budget <= 200 and not extra:
        expected_class, config_class = MultiObjectiveBayesian, MultiObjectiveBayesianConfig
    else:
        expected_class, config_class = NSGA2, NSGA2Config
    expected = config_class(objectives=objectives, max_evaluations=budget, seed=0)
    assert type(optimizer) is expected_class and optimizer.config.model_dump() == expected.model_dump()


def test_settings_and_starting_points_reach_the_chosen_optimizer():
    optimizer = choose_optimizer(NUMBERS, 500, seed=4, maximize=True, initial_points={"x": 1.0, "n": 2})
    assert optimizer.config.max_evaluations == 500 and optimizer.config.seed == 4 and optimizer.config.maximize
    assert optimizer.initial_points == [{"x": 1.0, "n": 2}]


def test_unknown_settings_are_refused():
    with pytest.raises(ValidationError):
        choose_optimizer(NUMBERS, 500, sead=4)


def test_the_chosen_optimizer_runs():
    result = choose_optimizer(NUMBERS, 30, seed=0).optimize(lambda x, n: (x - 1) ** 2 + (n - 3) ** 2)
    assert result.evaluations == 30 and result.best_params["n"] == 3


def picks_against_best(summaries, problems):
    """For each problem and budget in the saved summaries: the chosen optimizer's median error over the best one's."""
    errors = {(s["problem"], s["optimizer"], s["budget_per_dimension"]): s["median_error"] for s in summaries}
    ratios = {name: [] for name in ("choose_optimizer", "genetic", "cma-es", "bayesian")}
    for problem, per_dimension in sorted({(p, b) for p, _, b in errors}):
        available = {o: e for (p, o, b), e in errors.items() if p == problem and b == per_dimension}
        best = min(available.values())
        chosen = NAMES[type(choose_optimizer(problems[problem].search_space, per_dimension * problems[problem].dimensions))]
        for name in ratios:
            pick = chosen if name == "choose_optimizer" else name
            if pick in available:
                ratios[name].append((available[pick] + 1e-8) / (best + 1e-8))
    return {name: math.exp(statistics.mean(map(math.log, r))) for name, r in ratios.items()}


@pytest.mark.parametrize(
    "file, problems",
    [
        ("full-summaries.json", standard_problems),
        ("noisy-summaries.json", noisy_problems),
        ("ml-summaries.json", ml.ml_problems),
        ("constrained-summaries.json", constrained_problems),
        ("conditional-summaries.json", conditional_problems),
    ],
)
def test_its_picks_come_close_to_the_best_optimizer_on_the_benchmark(file, problems):
    """Kept true as the benchmark results are regenerated: the documented rule must stay backed by them."""
    summaries = json.loads((BENCHMARKS / file).read_text())
    ratios = picks_against_best(summaries, {p.name: p for p in problems()})
    assert ratios["choose_optimizer"] < 1.5, "within half again of the best optimizer's error, on geometric mean"
    assert ratios["choose_optimizer"] <= min(ratios[name] for name in ("genetic", "cma-es")), "better than always using one"
