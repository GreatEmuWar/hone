"""choose_optimizer: picking an optimizer by budget and parameters, and how well the picks do on the benchmark."""
import json
import math
import statistics
from pathlib import Path

import pytest
from pydantic import ValidationError

from hone import CMAES, BayesianOptimizer, GeneticOptimizer, choose_optimizer
from hone.benchmarks import ml, noisy_problems, standard_problems
from hone.parameters import ChoiceParameter, RangeParameter

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
    ],
)
def test_its_picks_come_close_to_the_best_optimizer_on_the_benchmark(file, problems):
    """Kept true as the benchmark results are regenerated: the documented rule must stay backed by them."""
    summaries = json.loads((BENCHMARKS / file).read_text())
    ratios = picks_against_best(summaries, {p.name: p for p in problems()})
    assert ratios["choose_optimizer"] < 1.5, "within half again of the best optimizer's error, on geometric mean"
    assert ratios["choose_optimizer"] <= min(ratios[name] for name in ("genetic", "cma-es")), "better than always using one"
