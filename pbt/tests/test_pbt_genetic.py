"""Property tests for hone.optimizers.genetic, from GeneticOptimizer's docstring and the README."""
import os
from contextlib import contextmanager

from hypothesis import HealthCheck, assume, given, settings, strategies as st

from hone import ChoiceParameter, GeneticOptimizer, GeneticOptimizerConfig, RangeParameter, SelectionMethod
from pbt_runs import Recorder, seeds
from pbt_strategies import search_spaces
from hone.optimizers.genetic import breeding as genetic

EXAMPLES = int(os.environ.get("PBT_EXAMPLES", "25"))
run_settings = settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow])
spaces = search_spaces(kinds=("range", "choice", "fixed", "choose_n"))


@contextmanager
def repeats_allowed():
    """Turn off repeat avoidance, which mutates a child that would repeat a point even with mutation off."""
    tries, genetic.REPEAT_TRIES = genetic.REPEAT_TRIES, 0
    try:
        yield
    finally:
        genetic.REPEAT_TRIES = tries


@st.composite
def configs(draw, budget=True, **fixed):
    settings_ = dict(
        seed=draw(seeds),
        mutation_rate=draw(st.floats(min_value=0, max_value=1)),
        crossover_rate=draw(st.floats(min_value=0, max_value=1)),
        elitism=draw(st.booleans()),
        elitism_rate=draw(st.floats(min_value=0, max_value=0.99)),
        selection_method=draw(st.sampled_from(list(SelectionMethod))),
        tournament_size=draw(st.integers(min_value=2, max_value=6)),
        tournament_p=draw(st.floats(min_value=0, max_value=1)),
        niche_share=draw(st.floats(min_value=0, max_value=0.99)),
        blend_crossover=draw(st.booleans()),
        blend_alpha=draw(st.floats(min_value=0, max_value=1)),
        maximize=draw(st.booleans()),
    )
    if budget:
        settings_["max_evaluations"] = draw(st.integers(min_value=10, max_value=150))
    if draw(st.booleans()):
        settings_["population_size"] = draw(st.integers(min_value=settings_["tournament_size"], max_value=30))
    settings_.update(fixed)
    return GeneticOptimizerConfig(**settings_)


def generations_of(optimizer, objective):
    """Drive the optimizer by ask and tell, returning each generation's trials with their values."""
    generations = []
    while not optimizer.done:
        trials = optimizer.ask()
        values = [objective(**t.params) for t in trials]
        for trial, value in zip(trials, values):
            optimizer.tell(trial, value)
        generations.append(list(zip(trials, values)))
    return generations


# README: "Given a budget, the genetic optimizer chooses its own population size ... and runs until the budget is
# spent, unless you set population_size or generations yourself." Here only population_size may be set; the
# generations default to the budget, so the budget is always spent.
@run_settings
@given(spaces, configs())
def test_runs_until_the_budget_is_spent(space, config):
    # An elitism_rate that rounded to the whole population once left nothing to evaluate; now at least one child is
    # bred each generation (see pbt/SUMMARY.md).
    objective = Recorder(space, sign=-1.0 if config.maximize else 1.0)
    result = GeneticOptimizer(space, config).optimize(objective)
    assert result.evaluations == config.max_evaluations


# Docstring: "The first generation is any initial points, then random candidates. Each later generation keeps the
# previous one's elites unchanged, then fills the rest of the population with children."
@run_settings
@given(spaces, configs(budget=False, generations=4))
def test_generation_sizes(space, config):
    generations = generations_of(GeneticOptimizer(space, config), Recorder(space))
    elites = min(round(config.population_size * config.elitism_rate), config.population_size - 1) if config.elitism else 0
    assert len(generations) == 4
    assert len(generations[0]) == config.population_size
    assert all(len(g) == config.population_size - elites for g in generations[1:])


# Docstring: children are made by crossover and mutation; with uniform crossover and no mutation every gene of a child
# comes from a parent,
# and "Failed evaluations are never elites or parents".
@run_settings
@given(spaces, configs(budget=False, generations=4, mutation_rate=0.0, blend_crossover=False), st.data())
def test_children_inherit_genes_from_successful_parents(space, config, data):
    with repeats_allowed():
        _children_inherit_genes_from_successful_parents(space, config, data)


def _children_inherit_genes_from_successful_parents(space, config, data):
    failing = {name for name in space.names if data.draw(st.booleans())}
    optimizer = GeneticOptimizer(space, config)
    successful = []  # every successfully evaluated point so far: parents and elites come from these
    first = True
    while not optimizer.done:
        trials = optimizer.ask()
        if not first and successful:
            for trial in trials:
                for name, value in trial.params.items():
                    assert any(value == p[name] for p in successful)
        for i, trial in enumerate(trials):
            if i % 3 == 0 and failing:
                optimizer.tell(trial, failed=True)
            else:
                optimizer.tell(trial, float(i))
                successful.append(trial.params)
        first = False


# A choice option whose every evaluation fails is never bred from, so without mutation it disappears after the
# first generation (unless every evaluation failed, when the population starts afresh).
@run_settings
@given(configs(budget=False, generations=5, mutation_rate=0.0), seeds)
def test_failed_options_are_not_bred(config, seed):
    with repeats_allowed():
        _failed_options_are_not_bred(config, seed)


def _failed_options_are_not_bred(config, seed):
    space = [ChoiceParameter("material", ["steel", "glass", "wood"]), RangeParameter("x", 0, 1)]
    optimizer = GeneticOptimizer(space, config)
    first = True
    while not optimizer.done:
        trials = optimizer.ask()
        if not first:
            assert all(t.params["material"] != "glass" for t in trials)
        all_failed = all(t.params["material"] == "glass" for t in trials)
        assume(not all_failed)
        for trial in trials:
            if trial.params["material"] == "glass":
                optimizer.tell(trial, failed=True)
            else:
                optimizer.tell(trial, trial.params["x"])
        first = False


# Elites are the best of the previous generation, carried over unchanged: so with elitism the best of each
# generation never gets worse.
@run_settings
@given(spaces, configs(budget=False, generations=6, elitism=True), st.booleans())
def test_elitism_keeps_the_best(space, config, _):
    assume(round(config.population_size * config.elitism_rate) >= 1)
    sign = -1.0 if config.maximize else 1.0
    result = GeneticOptimizer(space, config).optimize(Recorder(space, sign=sign))
    bests = [g.best_fitness for g in result.generations]
    assert bests == sorted(bests, reverse=not config.maximize)


# optimize(): "The genetic optimizer and random search return it however many workers are used".
@settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(configs(max_evaluations=40))
def test_workers_do_not_change_the_result(config):
    space = [RangeParameter("x", -5, 5), RangeParameter("y", -5, 5)]
    one = GeneticOptimizer(space, config).optimize(sphere)
    two = GeneticOptimizer(space, config).optimize(sphere, workers=2)
    assert one == two


def sphere(x, y):
    return x**2 + y**2
