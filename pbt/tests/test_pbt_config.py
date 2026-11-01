"""Property tests for hone.config, from its docstrings, comments and the README's Configuration section."""
import json

import pytest
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st
from pydantic import ValidationError

from hone.config import BayesianConfig, CMAESConfig, GeneticOptimizerConfig, OptimizerConfig, SelectionMethod

budgets = st.integers(min_value=1, max_value=100_000)


def optional(strategy):
    return st.none() | strategy


@st.composite
def genetic_configs(draw):
    kwargs = dict(
        max_evaluations=draw(optional(budgets)),
        maximize=draw(st.booleans()),
        threshold=draw(optional(st.floats(min_value=-1e6, max_value=1e6))),
        seed=draw(optional(st.integers(min_value=0, max_value=2**32 - 1))),
        noise=draw(st.booleans()),
        mutation_rate=draw(st.floats(min_value=0, max_value=1)),
        mutation_scale=draw(st.floats(min_value=1e-3, max_value=10)),
        crossover_rate=draw(st.floats(min_value=0, max_value=1)),
        elitism=draw(st.booleans()),
        elitism_rate=draw(st.floats(min_value=0, max_value=0.99)),
        selection_method=draw(st.sampled_from(list(SelectionMethod))),
        tournament_size=draw(st.integers(min_value=2, max_value=8)),
        tournament_p=draw(st.floats(min_value=0, max_value=1)),
        niche_share=draw(st.floats(min_value=0, max_value=0.99)),
        min_finished=draw(st.floats(min_value=0.01, max_value=1)),
    )
    if draw(st.booleans()):
        kwargs["population_size"] = draw(st.integers(min_value=kwargs["tournament_size"], max_value=200))
    if draw(st.booleans()):
        kwargs["generations"] = draw(st.integers(min_value=1, max_value=1000))
    budget = kwargs["max_evaluations"]
    if budget is None or budget > 1:
        kwargs["final_evaluations"] = draw(st.integers(min_value=0, max_value=(budget - 1) if budget else 100))
    return GeneticOptimizerConfig(**kwargs)


# Docstring and README: with a budget and no population_size, the population is about a twentieth of the
# budget, between 16 and 50 (and at least the tournament size), and generations default to as many as the budget allows.
@given(budgets, st.integers(min_value=2, max_value=60))
def test_genetic_population_from_budget(budget, tournament_size):
    config = GeneticOptimizerConfig(max_evaluations=budget, tournament_size=tournament_size)
    expected = max(min(max(budget // 20, 16), 50), tournament_size)
    assert config.population_size == expected
    assert config.generations * config.population_size >= budget  # enough generations to spend the budget


@given(budgets, st.integers(min_value=4, max_value=500), st.integers(min_value=1, max_value=500))
def test_genetic_explicit_settings_win(budget, population, generations):
    config = GeneticOptimizerConfig(max_evaluations=budget, population_size=population, generations=generations)
    assert (config.population_size, config.generations) == (population, generations)


def test_genetic_defaults_without_budget():
    config = GeneticOptimizerConfig()
    assert (config.population_size, config.generations) == (100, 25)


# BayesianConfig: "When max_evaluations is set and generations isn't, generations defaults to the budget."
@given(budgets)
def test_bayesian_generations_from_budget(budget):
    assert BayesianConfig(max_evaluations=budget).generations == budget


# tournament_size cannot exceed population_size.
@given(st.integers(min_value=2, max_value=100), st.integers(min_value=1, max_value=50))
def test_tournament_size_bounded_by_population(population, extra):
    with pytest.raises(ValidationError):
        GeneticOptimizerConfig(population_size=population, tournament_size=population + extra)


# final_evaluations must leave something of max_evaluations for the search.
@given(budgets, st.integers(min_value=0, max_value=1000))
def test_final_evaluations_leave_budget(budget, extra):
    with pytest.raises(ValidationError):
        OptimizerConfig(max_evaluations=budget, final_evaluations=budget + extra)


# reserved_evaluations: "Evaluations kept back for the final re-evaluation: none without noise."
@given(st.integers(min_value=0, max_value=1000), st.booleans())
def test_reserved_evaluations(final, noise):
    config = OptimizerConfig(final_evaluations=final, noise=noise)
    assert config.reserved_evaluations == (final if noise else 0)


# Configs are frozen.
@given(genetic_configs())
def test_configs_are_frozen(config):
    with pytest.raises(ValidationError):
        config.population_size = 5


# README: "It can also be loaded from a JSON or YAML file", and rebuild_optimizer re-validates the dumped config.
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(genetic_configs(), st.sampled_from(["json", "yaml", "yml"]))
def test_genetic_config_file_round_trip(tmp_path, config, suffix):
    data = config.model_dump(mode="json")
    path = tmp_path / f"config.{suffix}"
    path.write_text(json.dumps(data) if suffix == "json" else yaml.safe_dump(data))
    assert GeneticOptimizerConfig.from_file(path) == config
    assert GeneticOptimizerConfig.model_validate(data) == config


# Loading a file that gives only some settings applies the same defaults as the constructor.
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(optional(budgets), optional(st.integers(min_value=0, max_value=100)), st.sampled_from(["json", "yaml"]))
def test_partial_file_matches_constructor(tmp_path, budget, seed, suffix):
    data = {k: v for k, v in {"max_evaluations": budget, "seed": seed}.items() if v is not None}
    path = tmp_path / f"config.{suffix}"
    path.write_text(json.dumps(data) if suffix == "json" else yaml.safe_dump(data))
    for cls in (GeneticOptimizerConfig, BayesianConfig, CMAESConfig, OptimizerConfig):
        assert cls.from_file(path) == cls(**data)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.sampled_from([BayesianConfig, CMAESConfig, OptimizerConfig]), optional(budgets), st.booleans())
def test_other_config_round_trip(tmp_path, cls, budget, maximize):
    config = cls(max_evaluations=budget, maximize=maximize)
    assert cls.model_validate(config.model_dump(mode="json")) == config
    assert cls.model_validate_json(config.model_dump_json()) == config


# Unknown settings are refused (extra="forbid"), so a typo in a config file isn't silently ignored.
@given(st.sampled_from([GeneticOptimizerConfig, BayesianConfig, CMAESConfig, OptimizerConfig]))
def test_unknown_settings_refused(cls):
    with pytest.raises(ValidationError):
        cls(populaton_size=10)


# Field bounds documented by the Field constraints: min_finished in (0, 1], elitism_rate in [0, 1).
@given(st.one_of(st.floats(max_value=0, allow_nan=False), st.floats(min_value=1.0001, allow_nan=False)))
def test_min_finished_bounds(value):
    with pytest.raises(ValidationError):
        OptimizerConfig(min_finished=value)


@given(st.floats(min_value=1, allow_nan=False))
def test_elitism_rate_below_one(value):
    with pytest.raises(ValidationError):
        GeneticOptimizerConfig(elitism_rate=value)
