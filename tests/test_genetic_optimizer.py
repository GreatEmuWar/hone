"""How the genetic optimizer builds each generation."""
import pytest

from hone.parameters import FixedParameter, RangeParameter

from conftest import individual, make_optimizer


def evaluated_population(size):
    return [individual(float(i), x=float(i), y=0.0) for i in range(size)]


def test_first_generation_is_random_and_unevaluated():
    population = make_optimizer(population_size=10)._next_population(None, 0)
    assert len(population) == 10
    assert all(ind.fitness is None and ind.generation == 0 for ind in population)
    assert len({ind.params["x"] for ind in population}) == 10


@pytest.mark.parametrize("elitism", [True, False])
def test_population_size_is_constant(elitism):
    optimizer = make_optimizer(population_size=10, elitism=elitism, elitism_rate=0.3)
    assert len(optimizer._next_population(evaluated_population(10), 1)) == 10


def test_elites_carry_over_unchanged():
    previous = evaluated_population(10)
    population = make_optimizer(population_size=10, elitism_rate=0.3)._next_population(previous, 1)
    assert population[:3] == previous[:3]
    assert all(ind.fitness is not None for ind in population[:3])
    assert all(ind.fitness is None and ind.generation == 1 for ind in population[3:])


def test_maximizing_keeps_the_highest_as_elites():
    previous = evaluated_population(10)
    population = make_optimizer(population_size=10, elitism_rate=0.2, maximize=True)._next_population(previous, 1)
    assert [ind.fitness for ind in population[:2]] == [9.0, 8.0]


def test_no_elitism_carries_nobody_over():
    previous = evaluated_population(10)
    population = make_optimizer(population_size=10, elitism=False)._next_population(previous, 1)
    assert not any(ind in previous for ind in population)
    assert all(ind.fitness is None for ind in population)


def test_elites_are_not_evaluated_again():
    calls = []

    def counted(x, y):
        calls.append(1)
        return x**2 + y**2

    result = make_optimizer(objective_function=counted, generations=2, population_size=10, elitism_rate=0.3).optimize()
    assert len(calls) == result.evaluations == 10 + 7


def test_children_keep_fixed_parameters():
    optimizer = make_optimizer(
        search_space=[RangeParameter("x", -5, 5), FixedParameter("y", 3)], mutation_rate=1.0, population_size=10
    )
    previous = [individual(float(i), x=float(i), y=3) for i in range(10)]
    assert all(ind.params["y"] == 3 for ind in optimizer._next_population(previous, 1))


@pytest.mark.parametrize("selection_method", ["tournament", "roulette"])
def test_both_selection_methods_run(selection_method):
    assert make_optimizer(selection_method=selection_method, generations=3, population_size=10).optimize()


def test_readme_example_finds_minimum():
    """Mirrors the README: default config, minimizing x**2 + y**2 within [-5, 5]."""
    assert make_optimizer().optimize().best_fitness < 0.1


@pytest.mark.parametrize("seed", range(5))
def test_converges_on_sphere_across_seeds(seed):
    """Guards against the algorithm stalling near its best initial individual."""
    assert make_optimizer(seed=seed).optimize().best_fitness < 0.01
