import pytest
from pydantic import ValidationError

from hone import Result
from hone.parameters import ChoiceParameter, RangeParameter

from conftest import run


def test_save_and_load_round_trip(tmp_path):
    space = [RangeParameter("x", -5, 5, integer=True), ChoiceParameter("y", [0.5, 2.0])]
    result = run(search_space=space, generations=3, population_size=10)
    path = tmp_path / "run.json"
    result.save(path)
    assert Result.load(path) == result


def test_result_is_immutable():
    result = run(generations=1, population_size=4)
    with pytest.raises(ValidationError):
        result.best_fitness = 0


def test_saved_result_is_readable_json(tmp_path):
    path = tmp_path / "run.json"
    run(generations=2, population_size=4).save(path)
    text = path.read_text()
    for field in ("best_fitness", "best_params", "generations", "evaluations", "reached_threshold"):
        assert f'"{field}"' in text
