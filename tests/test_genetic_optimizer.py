import math

import pytest

from hone.constraint import Constraint
from hone.optimizers import Generation, Individual, Population

from conftest import individual, make_optimizer


def generation_of(*fitnesses):
    return Generation(0, population=Population(individuals=[individual(f, x=f) for f in fitnesses]))


# --- parameter sampling ---------------------------------------------------


@pytest.mark.xfail(reason="bug 1: optimizer reads Constraint.param, class defines param_name")
def test_random_parameters_respect_bounds():
    optimizer = make_optimizer()
    for _ in range(50):
        params = optimizer._random_parameters()
        assert -5 <= params["x"] <= 5
        assert -5 <= params["y"] <= 5


@pytest.mark.xfail(reason="bug 1: optimizer reads Constraint.param, class defines param_name")
def test_random_parameters_literal_and_choices():
    optimizer = make_optimizer(
        constraints=[
            Constraint(param_name="x", literal=3.0),
            Constraint(param_name="y", choices=[1, 2, 3]),
        ]
    )
    params = optimizer._random_parameters()
    assert params["x"] == 3.0
    assert params["y"] in (1, 2, 3)


@pytest.mark.xfail(reason="bug 3: type check compares str(type(value)) to 'float'")
def test_random_parameters_without_constraints_perturb_seed():
    optimizer = make_optimizer(constraints=[], seed_parameters={"x": 1.0, "y": 2})
    params = optimizer._random_parameters()
    assert abs(params["x"] - 1.0) <= 1
    assert isinstance(params["y"], int)


@pytest.mark.xfail(reason="bug 4: a constraint with only one bound passes None to uniform")
def test_random_parameters_with_one_bound():
    optimizer = make_optimizer(constraints=[Constraint(param_name="x", minimum=0)], seed_parameters={"x": 1.0})
    assert optimizer._random_parameters()["x"] >= 0


# --- crossover and mutation -----------------------------------------------


def test_crossover_takes_each_gene_from_a_parent():
    optimizer = make_optimizer()
    p1, p2 = Individual(x=1, y=2), Individual(x=10, y=20)
    for _ in range(20):
        child = optimizer._single_point_crossover(p1, p2)
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


@pytest.mark.xfail(reason="bug 10: roulette ignores the optimization direction")
def test_roulette_favours_lower_fitness_when_minimizing():
    optimizer = make_optimizer(selection_method="roulette", maximize=False)
    generation = generation_of(1, 99)
    picks = [optimizer._select_individual(generation).fitness for _ in range(200)]
    assert picks.count(1) > 150


@pytest.mark.xfail(reason="bug 10: roulette breaks on negative fitness")
def test_roulette_handles_negative_fitness():
    optimizer = make_optimizer(selection_method="roulette", maximize=True)
    optimizer._select_individual(generation_of(-5, 1, 3))


def test_unknown_selection_method_raises():
    optimizer = make_optimizer(selection_method="foo")
    with pytest.raises(ValueError):
        optimizer._select_individual(generation_of(1, 2))


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


@pytest.mark.xfail(reason="bugs 1, 2: constraint field mismatch; threshold=None crashes")
def test_optimize_runs_with_default_threshold():
    result = make_optimizer(generations=3, population_size=10).optimize()
    assert len(result.generations) == 3


@pytest.mark.xfail(reason="bugs 1, 8: population grows by one after the first generation")
def test_population_size_is_constant():
    result = make_optimizer(generations=4, population_size=10, threshold=float("inf")).optimize()
    assert [len(g.population) for g in result.generations] == [10] * 4


@pytest.mark.xfail(reason="bugs 1, 5, 6: minimizing never updates the optimum; best uses max()")
def test_minimizing_reports_lowest_fitness():
    result = make_optimizer(generations=5, population_size=20, maximize=False, threshold=-1).optimize()
    lowest = min(g.best.fitness for g in result.generations)
    assert result.best_fitness == lowest
    assert result.optimum == lowest


@pytest.mark.xfail(reason="bugs 1, 7: Result.best is an id string, not an Individual")
def test_result_best_is_an_individual():
    result = make_optimizer(generations=2, population_size=10, threshold=float("inf")).optimize()
    assert isinstance(result.best, Individual)


@pytest.mark.xfail(reason="bugs 1, 2, 9: README example cannot run, and the default maximizes")
def test_readme_example_finds_minimum():
    """Mirrors the README: default config, minimizing x**2 + y**2 within [-5, 5]."""
    result = make_optimizer().optimize()
    assert result.optimum < 0.1
