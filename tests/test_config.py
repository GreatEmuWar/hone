import json

import pytest
import yaml
from pydantic import ValidationError

from hone.config import GeneticOptimizerConfig, OptimizerConfig, SelectionMethod


def test_defaults():
    config = GeneticOptimizerConfig()
    assert config.population_size == 100
    assert config.generations == 25
    assert config.selection_method is SelectionMethod.TOURNAMENT
    assert config.maximize is False
    assert config.threshold is None
    assert config.seed is None


@pytest.mark.parametrize("budget, population", [(20, 16), (320, 16), (400, 20), (660, 33), (1000, 50), (100_000, 50)])
def test_a_budget_sizes_the_population(budget, population):
    config = GeneticOptimizerConfig(max_evaluations=budget)
    assert config.population_size == population
    assert config.generations == budget, "the budget, not the generation count, ends the run"


def test_settings_given_with_a_budget_are_kept():
    config = GeneticOptimizerConfig(max_evaluations=1000, population_size=20, generations=7)
    assert (config.population_size, config.generations) == (20, 7)


def test_a_budget_sized_population_fits_the_tournament():
    assert GeneticOptimizerConfig(max_evaluations=50, tournament_size=20).population_size == 20


def test_budget_defaults_apply_to_files_too(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"max_evaluations": 600}))
    assert GeneticOptimizerConfig.from_file(path).population_size == 30


def test_from_dict():
    config = GeneticOptimizerConfig.model_validate({"population_size": 10, "selection_method": "roulette"})
    assert config.population_size == 10
    assert config.selection_method is SelectionMethod.ROULETTE


def test_from_json_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"generations": 7}))
    assert GeneticOptimizerConfig.from_file(path).generations == 7


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_from_yaml_file(tmp_path, suffix):
    path = tmp_path / f"config{suffix}"
    path.write_text(yaml.safe_dump({"generations": 7, "selection_method": "roulette"}))
    config = GeneticOptimizerConfig.from_file(path)
    assert config.generations == 7
    assert config.selection_method is SelectionMethod.ROULETTE


def test_empty_yaml_file_gives_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    assert GeneticOptimizerConfig.from_file(path) == GeneticOptimizerConfig()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        GeneticOptimizerConfig.from_file(tmp_path / "missing.json")


def test_invalid_file_contents_raise(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"generations": 0}))
    with pytest.raises(ValidationError):
        GeneticOptimizerConfig.from_file(path)


@pytest.mark.parametrize(
    "fields",
    [
        {"generatoins": 7},
        {"selection_method": "foo"},
        {"mutation_rate": 1.5},
        {"crossover_rate": -0.1},
        {"mutation_scale": 0},
        {"mutation_threshold": 0.1},
        {"elitism_rate": 1.0},
        {"population_size": 1},
        {"tournament_size": 1},
        {"population_size": 4, "tournament_size": 5},
    ],
)
def test_invalid_values_rejected(fields):
    with pytest.raises(ValidationError):
        GeneticOptimizerConfig(**fields)


def test_config_is_immutable():
    with pytest.raises(ValidationError):
        GeneticOptimizerConfig().generations = 3


def test_base_config_loads_from_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"generations": 7, "maximize": True}))
    config = OptimizerConfig.from_file(path)
    assert config == OptimizerConfig(generations=7, maximize=True)


def test_base_config_rejects_genetic_settings():
    with pytest.raises(ValidationError):
        OptimizerConfig(mutation_rate=0.2)


def test_genetic_config_extends_base_config():
    assert isinstance(GeneticOptimizerConfig(), OptimizerConfig)
