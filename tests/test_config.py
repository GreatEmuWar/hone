import json

import pytest
import yaml

from hone.config import GeneticOptimizerConfig


def test_defaults():
    config = GeneticOptimizerConfig()
    assert config.population_size == 100
    assert config.generations == 25
    assert config.selection_method == "tournament"
    assert config.threshold is None


def test_from_dict():
    config = GeneticOptimizerConfig.from_dict({"population_size": 10, "maximize": False})
    assert config.population_size == 10
    assert config.maximize is False


def test_from_jsonfile(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"generations": 7}))
    assert GeneticOptimizerConfig.from_jsonfile(path).generations == 7


def test_from_yamlfile(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"generations": 7, "selection_method": "roulette"}))
    config = GeneticOptimizerConfig.from_yamlfile(path)
    assert config.generations == 7
    assert config.selection_method == "roulette"


@pytest.mark.xfail(reason="bug 14: loaders print the error and return None")
def test_from_dict_rejects_unknown_keys():
    with pytest.raises(TypeError):
        GeneticOptimizerConfig.from_dict({"generatoins": 7})


@pytest.mark.xfail(reason="bug 14: loaders print the error and return None")
def test_from_jsonfile_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        GeneticOptimizerConfig.from_jsonfile(tmp_path / "missing.json")


@pytest.mark.xfail(reason="bug 14: config values are not validated")
def test_invalid_selection_method_rejected():
    with pytest.raises(ValueError):
        GeneticOptimizerConfig(selection_method="foo")


@pytest.mark.xfail(reason="bug 14: config values are not validated")
def test_rate_out_of_range_rejected():
    with pytest.raises(ValueError):
        GeneticOptimizerConfig(mutation_rate=1.5)
