"""How the genetic optimizer builds each generation."""
import pytest

from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter

from conftest import individual, make_optimizer, run


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

    result = run(objective_function=counted, generations=2, population_size=10, elitism_rate=0.3)
    assert len(calls) == result.evaluations == 10 + 7


def test_children_keep_fixed_parameters():
    optimizer = make_optimizer(
        search_space=[RangeParameter("x", -5, 5), FixedParameter("y", 3)], mutation_rate=1.0, population_size=10
    )
    previous = [individual(float(i), x=float(i), y=3) for i in range(10)]
    assert all(ind.params["y"] == 3 for ind in optimizer._next_population(previous, 1))


@pytest.mark.parametrize("selection_method", ["tournament", "roulette"])
def test_both_selection_methods_run(selection_method):
    assert run(selection_method=selection_method, generations=3, population_size=10)


def test_readme_example_finds_minimum():
    """Mirrors the README: default config, minimizing x**2 + y**2 within [-5, 5]."""
    assert run().best_fitness < 0.1


@pytest.mark.parametrize("seed", range(5))
def test_converges_on_sphere_across_seeds(seed):
    """Guards against the algorithm stalling near its best initial individual."""
    assert run(seed=seed).best_fitness < 0.01


# --- niches ---------------------------------------------------------------

NICHE_SPACE = [RangeParameter("x", -5, 5), ChoiceParameter("material", ["steel", "aluminium", "titanium"])]


def niche_population():
    """Steel is fittest by far; aluminium and titanium are each held by one weak individual."""
    steel = [individual(float(i), x=float(i) / 10, material="steel") for i in range(8)]
    return steel + [individual(50.0, x=1.0, material="aluminium"), individual(60.0, x=2.0, material="titanium")]


def test_niches_keep_every_option_breeding():
    optimizer = make_optimizer(search_space=NICHE_SPACE, population_size=20, niche_share=0.5, elitism=False)
    children = optimizer._next_population(niche_population(), 1)
    niched = children[:10]
    assert {c.params["material"] for c in niched} == {"steel", "aluminium", "titanium"}
    assert sum(c.params["material"] == "titanium" for c in niched) == 3, "an equal share per niche"


def test_niche_children_take_their_niches_options_even_when_mutated():
    optimizer = make_optimizer(search_space=NICHE_SPACE, population_size=20, mutation_rate=1.0)
    niched = optimizer._niche_children(niche_population(), 6, 1)
    assert [c.params["material"] for c in niched] == ["steel"] * 2 + ["aluminium"] * 2 + ["titanium"] * 2, "fittest niche first"


def test_without_niches_the_weak_options_are_rarely_bred():
    optimizer = make_optimizer(search_space=NICHE_SPACE, population_size=20, mutation_rate=0.0, elitism=False, seed=1)
    children = optimizer._next_population(niche_population(), 1)
    assert sum(c.params["material"] == "titanium" for c in children) <= 2


def test_niches_need_a_choice_parameter():
    optimizer = make_optimizer(niche_share=0.5, population_size=10)
    assert optimizer._niche_children([individual(1.0, x=0.0, y=0.0)], 5, 1) == []


def test_more_niches_than_children_breed_the_fittest():
    optimizer = make_optimizer(search_space=NICHE_SPACE, population_size=20, niche_share=0.5)
    niched = optimizer._niche_children(niche_population(), 2, 1)
    assert [c.params["material"] for c in niched] == ["steel", "aluminium"]


def test_niches_solve_the_categorical_problem_the_default_misses():
    """On material-selection-10d the default often settles on the wrong material; niches don't."""
    from hone import GeneticOptimizer, GeneticOptimizerConfig
    from hone.benchmarks import standard_problems

    problem = {p.name: p for p in standard_problems()}["material-selection-10d"]

    def materials(**settings):
        config = lambda seed: GeneticOptimizerConfig(max_evaluations=10000, seed=seed, **settings)  # noqa: E731
        return [GeneticOptimizer(problem.search_space, config(seed)).optimize(problem.function).best_params["material"] for seed in range(6)]

    assert materials(niche_share=0.5) == ["titanium"] * 6
    assert materials().count("titanium") < 6
