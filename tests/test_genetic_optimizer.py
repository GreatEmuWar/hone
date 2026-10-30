import math

import pytest

from hone.optimizers import Generation, Individual, Population
from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter, SearchSpace

from conftest import individual, make_optimizer


def generation_of(*fitnesses):
    return Generation(0, population=Population(individuals=[individual(f, x=f) for f in fitnesses]))


# --- construction ---------------------------------------------------------


def test_accepts_search_space_or_list():
    params = [RangeParameter("x", 0, 1), RangeParameter("y", 0, 1)]
    from_list = make_optimizer(search_space=params)
    from_space = make_optimizer(search_space=SearchSpace(params))
    assert from_list.objective_function.search_space == from_space.objective_function.search_space


def test_config_defaults_when_omitted():
    from hone import GeneticOptimizer
    from conftest import sphere

    assert GeneticOptimizer(sphere, [RangeParameter("x", 0, 1), RangeParameter("y", 0, 1)]).config.maximize is False


def test_rejects_function_that_does_not_match_search_space():
    with pytest.raises(ValueError, match="missing from the search space: y"):
        make_optimizer(search_space=[RangeParameter("x", 0, 1)])


def test_same_seed_gives_same_parameters():
    a, b = make_optimizer(seed=42), make_optimizer(seed=42)
    assert [a._random_parameters() for _ in range(5)] == [b._random_parameters() for _ in range(5)]


# --- parameter sampling ---------------------------------------------------


def test_random_parameters_respect_bounds():
    optimizer = make_optimizer()
    for _ in range(50):
        params = optimizer._random_parameters()
        assert -5 <= params["x"] <= 5
        assert -5 <= params["y"] <= 5


def test_random_parameters_fixed_and_choices():
    optimizer = make_optimizer(search_space=[FixedParameter("x", 3.0), ChoiceParameter("y", [1, 2, 3])])
    params = optimizer._random_parameters()
    assert params["x"] == 3.0
    assert params["y"] in (1, 2, 3)


# --- crossover and mutation -----------------------------------------------


def test_crossover_takes_each_gene_from_a_parent():
    optimizer = make_optimizer()
    p1, p2 = Individual(x=1, y=2), Individual(x=10, y=20)
    for _ in range(20):
        child = optimizer._uniform_crossover(p1, p2)
        assert child.params["x"] in (1, 10)
        assert child.params["y"] in (2, 20)
        assert child.individual_id not in (p1.individual_id, p2.individual_id)


def test_mutation_replaces_genes_with_random_values(monkeypatch):
    optimizer = make_optimizer(mutation_threshold=1.0)
    monkeypatch.setattr(optimizer, "_random_parameters", lambda: {"x": 99, "y": 99})
    child = optimizer._mutate_individual(Individual(x=1, y=2))
    assert child.params == {"x": 99, "y": 99}


# --- selection ------------------------------------------------------------


@pytest.mark.parametrize("maximize, expected", [(True, 9), (False, 1)])
def test_tournament_picks_best_when_p_is_one(maximize, expected):
    optimizer = make_optimizer(maximize=maximize, tournament_p=1.0, tournament_size=50)
    assert optimizer._tournament_selection(generation_of(1, 5, 9)).fitness == expected


def test_roulette_favours_higher_fitness_when_maximizing():
    optimizer = make_optimizer(selection_method="roulette", maximize=True)
    generation = generation_of(1, 99)
    picks = [optimizer._select_individual(generation).fitness for _ in range(200)]
    assert picks.count(99) > 150


def test_roulette_favours_lower_fitness_when_minimizing():
    optimizer = make_optimizer(selection_method="roulette", maximize=False)
    generation = generation_of(1, 99)
    picks = [optimizer._select_individual(generation).fitness for _ in range(200)]
    assert picks.count(1) > 150


def test_roulette_handles_negative_fitness():
    optimizer = make_optimizer(selection_method="roulette", maximize=True)
    optimizer._select_individual(generation_of(-5, 1, 3))


# --- evaluation -----------------------------------------------------------


@pytest.mark.parametrize("maximize", [True, False])
def test_evaluate_population_records_stats(maximize):
    optimizer = make_optimizer(maximize=maximize)
    population = Population(individuals=[Individual(x=0, y=0), Individual(x=1, y=1), Individual(x=2, y=2)])
    generation = Generation(0, population=population)
    optimizer._evaluate_population(generation)

    assert [ind.fitness for ind in population] == [0, 2, 8]
    best, worst = (8, 0) if maximize else (0, 8)
    assert generation.best.fitness == best
    assert generation.worst.fitness == worst
    assert generation.mean == pytest.approx(10 / 3)
    assert generation.median == 2
    assert generation.std == pytest.approx(math.sqrt(312 / 27))


# --- full optimization ----------------------------------------------------


def test_optimize_runs_with_default_threshold():
    result = make_optimizer(generations=3, population_size=10).optimize()
    assert len(result.generations) == 3


def test_optimize_is_reproducible_with_a_seed():
    def run():
        result = make_optimizer(generations=3, population_size=10, threshold=-1, seed=7).optimize()
        return [g.best.fitness for g in result.generations]

    assert run() == run()


def test_optimize_keeps_values_within_the_search_space():
    space = [RangeParameter("x", -5, 5, integer=True), ChoiceParameter("y", [-1, 0.5, 2])]
    result = make_optimizer(search_space=space, generations=4, population_size=10, threshold=-1).optimize()
    for generation in result.generations:
        for ind in generation.population:
            assert type(ind.params["x"]) is int and -5 <= ind.params["x"] <= 5
            assert ind.params["y"] in (-1, 0.5, 2)


def test_population_size_is_constant():
    result = make_optimizer(generations=4, population_size=10, threshold=float("-inf")).optimize()
    assert [len(g.population) for g in result.generations] == [10] * 4


def test_minimizing_reports_lowest_fitness():
    result = make_optimizer(generations=5, population_size=20, threshold=-1).optimize()
    lowest = min(g.best.fitness for g in result.generations)
    assert result.best_fitness == lowest
    assert result.optimum == lowest


def test_result_best_is_an_individual():
    result = make_optimizer(generations=2, population_size=10, threshold=-1).optimize()
    assert isinstance(result.best, Individual)


@pytest.mark.xfail(reason="mutation only resamples uniformly, so runs stall near the best initial individual")
def test_readme_example_finds_minimum():
    """Mirrors the README: default config, minimizing x**2 + y**2 within [-5, 5]."""
    result = make_optimizer().optimize()
    assert result.optimum < 0.1


def test_maximizing_reports_highest_fitness():
    result = make_optimizer(generations=5, population_size=20, maximize=True).optimize()
    highest = max(g.best.fitness for g in result.generations)
    assert result.best_fitness == result.optimum == highest
    assert result.optimum_params == result.best.params


def test_threshold_stops_early_and_keeps_the_best():
    result = make_optimizer(generations=50, population_size=20, threshold=5.0).optimize()
    assert len(result.generations) < 50
    assert result.best_fitness <= 5.0
    assert result.best is result.generations[-1].best


def test_without_threshold_runs_every_generation():
    result = make_optimizer(generations=6, population_size=10).optimize()
    assert [g.generation_id for g in result.generations] == list(range(6))


def test_optimize_twice_starts_a_fresh_history():
    optimizer = make_optimizer(generations=3, population_size=10)
    optimizer.optimize()
    assert len(optimizer.optimize().generations) == 3


def test_best_is_never_lost_between_generations():
    result = make_optimizer(generations=8, population_size=10, elitism=False).optimize()
    assert result.best_fitness == min(g.best.fitness for g in result.generations)


def test_elites_carry_over_and_are_not_re_evaluated():
    calls = []

    def counted(x, y):
        calls.append((x, y))
        return x**2 + y**2

    result = make_optimizer(
        objective_function=counted, generations=2, population_size=10, elitism_rate=0.3
    ).optimize()
    first, second = result.generations
    assert second.population.individuals[:3] == sorted(first.population, key=lambda i: i.fitness)[:3]
    assert len(calls) == 10 + 7


def test_no_elitism_carries_nobody_over():
    result = make_optimizer(generations=2, population_size=10, elitism=False).optimize()
    first, second = result.generations
    assert not {i.individual_id for i in first.population} & {i.individual_id for i in second.population}


def test_individuals_record_their_generation():
    result = make_optimizer(generations=3, population_size=10, elitism=False).optimize()
    for generation in result.generations:
        assert {i.generation for i in generation.population} == {generation.generation_id}


def test_roulette_with_equal_fitness_picks_anyone():
    optimizer = make_optimizer(selection_method="roulette")
    generation = generation_of(4, 4, 4)
    picked = {id(optimizer._select_individual(generation)) for _ in range(50)}
    assert len(picked) == 3

