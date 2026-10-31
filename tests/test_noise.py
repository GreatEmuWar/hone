"""Noisy objectives: averaging repeated evaluations, the final round, and the noisy benchmark."""
import numpy as np
import pytest
from pydantic import ValidationError

from hone import BayesianConfig, BayesianOptimizer, CMAES, CMAESConfig, GeneticOptimizer, GeneticOptimizerConfig, RandomSearch
from hone.benchmarks import noisy_problems, run_benchmark
from hone.benchmarks.__main__ import main
from hone.config import OptimizerConfig
from hone.optimizers.base import Individual, _key
from hone.parameters import ChoiceParameter, RangeParameter

from conftest import bounded_parameters

SPACE = [RangeParameter("x", -5, 5), RangeParameter("y", -5, 5)]


def random_search(**config):
    config.setdefault("seed", 0)
    return RandomSearch(SPACE, OptimizerConfig(**config))


def tell_values(optimizer, values):
    """Tell each trial of the next ask() the next value from values."""
    trials = optimizer.ask()
    for trial, value in zip(trials, values):
        optimizer.tell(trial, value)
    return trials


# --- configuration --------------------------------------------------------


def test_final_evaluations_need_noise_to_be_kept_back():
    assert OptimizerConfig(final_evaluations=10).reserved_evaluations == 0
    assert OptimizerConfig(noise=True, final_evaluations=10).reserved_evaluations == 10
    assert OptimizerConfig(noise=True).reserved_evaluations == 0


def test_final_evaluations_must_leave_a_budget_for_the_search():
    with pytest.raises(ValidationError, match="leaves nothing"):
        OptimizerConfig(noise=True, max_evaluations=10, final_evaluations=10)


def test_keys_group_equal_points_including_lists():
    assert _key({"x": 1.0, "tags": ["a", "b"]}) == _key({"x": 1.0, "tags": ["a", "b"]})
    assert _key({"x": 1.0, "tags": ["a", "b"]}) != _key({"x": 1.0, "tags": ["b", "a"]})


# --- averaging ------------------------------------------------------------


def test_repeated_points_are_averaged():
    optimizer = RandomSearch([RangeParameter("n", 0, 0, integer=True)], OptimizerConfig(population_size=3, generations=1, noise=True))
    tell_values(optimizer, [1.0, 2.0, 6.0])
    result = optimizer.result()
    assert result.best_fitness == 3.0 and result.best_params == {"n": 0}


def test_without_noise_the_luckiest_value_counts():
    optimizer = RandomSearch([RangeParameter("n", 0, 0, integer=True)], OptimizerConfig(population_size=3, generations=1))
    tell_values(optimizer, [1.0, 2.0, 6.0])
    assert optimizer.result().best_fitness == 1.0


def test_failures_are_not_averaged_in():
    optimizer = RandomSearch([RangeParameter("n", 0, 0, integer=True)], OptimizerConfig(population_size=3, generations=1, noise=True))
    first, second, third = optimizer.ask()
    optimizer.tell(first, 2.0)
    optimizer.tell(second, failed=True)
    optimizer.tell(third, 4.0)
    assert optimizer.result().best_fitness == 3.0 and optimizer.result().failures == 1


def test_genetic_parents_are_ranked_by_their_latest_average():
    optimizer = GeneticOptimizer(
        [RangeParameter("n", 0, 0, integer=True)],
        GeneticOptimizerConfig(population_size=4, tournament_size=2, generations=3, noise=True, seed=0),
    )
    tell_values(optimizer, [0.0, 10.0, 10.0, 10.0])
    first = optimizer._population[0]
    assert first.fitness == 0.0, "its own evaluation, before the others of the same point arrived"
    optimizer.ask()
    assert first.fitness == 7.5, "refreshed to the average of all four when the next generation is bred"


# --- the final round ------------------------------------------------------


def test_final_round_re_evaluates_the_most_promising_points():
    optimizer = random_search(population_size=10, max_evaluations=30, noise=True, final_evaluations=10)
    while not optimizer.done:
        for trial in optimizer.ask():
            optimizer.tell(trial, trial.params["x"] ** 2 + trial.params["y"] ** 2)
    result = optimizer.result()
    assert result.evaluations == 30
    final = optimizer._population
    assert len(final) == 10 and len({_key(i.params) for i in final}) == 5, "5 finalists, each evaluated twice more"
    assert _key(result.best_params) in optimizer._finalists
    assert result.generations[-1].evaluations == 30


def test_final_round_chooses_by_the_average_not_the_lucky_value():
    space = [ChoiceParameter("design", ["steady", "lucky"])]
    values = {"steady": iter([1.0, 1.0, 1.0, 1.0]), "lucky": iter([0.5, 3.0, 3.0, 3.0])}
    optimizer = RandomSearch(space, OptimizerConfig(population_size=2, max_evaluations=6, noise=True, final_evaluations=4, seed=0))
    # The first generation holds one of each design.
    designs = iter(["steady", "lucky"])
    optimizer._random_individual = lambda generation: Individual(params={"design": next(designs)}, generation=generation)
    while not optimizer.done:
        for trial in optimizer.ask():
            optimizer.tell(trial, next(values[trial.params["design"]]))
    result = optimizer.result()
    assert result.best_params == {"design": "steady"} and result.best_fitness == 1.0


def test_final_round_runs_after_the_last_generation_without_a_budget():
    optimizer = random_search(population_size=5, generations=2, noise=True, final_evaluations=4)
    while not optimizer.done:
        for trial in optimizer.ask():
            optimizer.tell(trial, trial.params["x"] ** 2)
    assert optimizer.result().evaluations == 14 and len(optimizer.result().generations) == 3


@pytest.mark.parametrize(
    "make",
    [
        lambda: GeneticOptimizer(bounded_parameters(), GeneticOptimizerConfig(max_evaluations=100, noise=True, final_evaluations=10, seed=0)),
        lambda: CMAES(bounded_parameters(), CMAESConfig(max_evaluations=100, noise=True, final_evaluations=10, seed=0)),
        lambda: BayesianOptimizer(bounded_parameters(), BayesianConfig(max_evaluations=30, noise=True, final_evaluations=6, seed=0)),
    ],
    ids=["genetic", "cma-es", "bayesian"],
)
def test_every_optimizer_runs_the_final_round_within_its_budget(make):
    rng = np.random.default_rng(0)
    optimizer = make()
    result = optimizer.optimize(lambda x, y: x**2 + y**2 + rng.normal())
    assert result.evaluations == optimizer.config.max_evaluations
    assert _key(result.best_params) in optimizer._finalists


def test_noise_handling_finds_better_points_on_a_noisy_function():
    """On integers the genetic optimizer revisits points often, and averaging stops a lucky one winning."""
    space = [RangeParameter("a", -10, 10, integer=True), RangeParameter("b", -10, 10, integer=True)]

    def true(a, b):
        return (a - 3) ** 2 + (b + 2) ** 2

    errors = {False: [], True: []}
    for noise in errors:
        for seed in range(20):
            rng = np.random.default_rng(seed)
            config = GeneticOptimizerConfig(max_evaluations=800, noise=noise, seed=seed)
            result = GeneticOptimizer(space, config).optimize(lambda a, b: true(a, b) + 3 * rng.normal())
            errors[noise].append(true(**result.best_params))
    assert np.mean(errors[True]) < 0.5 * np.mean(errors[False])


# --- the noisy benchmark --------------------------------------------------


def test_noisy_problems_add_noise_but_score_the_true_value():
    problems = noisy_problems()
    assert all(p.noise > 0 and p.name.startswith("noisy-") for p in problems)
    runs = run_benchmark(problems[:1], seeds=range(2), budgets_per_dimension=[4])
    assert runs == run_benchmark(problems[:1], seeds=range(2), budgets_per_dimension=[4]), "the noise is repeatable"
    for run in runs:
        assert run.error >= 0 and run.trace[-1][1] >= 0


def test_noisy_runs_are_scored_by_the_true_value_of_the_chosen_point():
    problem = noisy_problems()[0]
    chosen = {}

    def recording(problem, budget, seed):
        optimizer = RandomSearch(problem.search_space, OptimizerConfig(population_size=10, max_evaluations=budget, seed=seed, noise=True))
        original = optimizer.result
        optimizer.result = lambda: chosen.setdefault("result", original())
        return optimizer

    run = run_benchmark([problem], optimizers={"r": recording}, seeds=range(1), budgets_per_dimension=[4])[0]
    result = chosen["result"]
    assert run.error == pytest.approx(problem.function(**result.best_params) - problem.optimum)
    assert run.error != pytest.approx(max(result.best_fitness - problem.optimum, 0)), "the noisy value would score it differently"


def test_cli_runs_the_noisy_suite(tmp_path):
    results = tmp_path / "noisy.md"
    assert main(["--noisy", "--seeds", "1", "--budgets", "2", "--markdown", str(results)]) == 0
    assert "| noisy-sphere-5d |" in results.read_text()


def test_the_threshold_is_checked_against_the_current_average():
    """One lucky value doesn't stop the run once later evaluations of that point bring its average back up."""
    config = OptimizerConfig(population_size=4, noise=True, threshold=-0.5, seed=0)
    optimizer = RandomSearch([RangeParameter("n", 0, 1, integer=True)], config)
    trials = optimizer.ask()
    assert [t.params["n"] for t in trials] == [1, 1, 1, 0]
    for trial, value in zip(trials, [-1.0, 1.0, 1.0, 1.0]):
        optimizer.tell(trial, value)  # n=1 is lucky once, then averages 1/3
    result = optimizer.result()
    assert not result.reached_threshold and not optimizer.done and result.best_fitness == pytest.approx(1 / 3)
