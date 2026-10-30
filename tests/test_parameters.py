import numpy as np
import pytest
from pydantic import ValidationError

from hone.parameters import ChoiceParameter, FixedParameter, RangeParameter, SearchSpace


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_range_samples_within_bounds(rng):
    x = RangeParameter("x", -5, 5)
    samples = [x.sample(rng) for _ in range(200)]
    assert all(-5 <= s <= 5 for s in samples)
    assert all(isinstance(s, float) for s in samples)


def test_integer_range_samples_integers_including_bounds(rng):
    n = RangeParameter("n", 1, 3, integer=True)
    samples = {n.sample(rng) for _ in range(200)}
    assert samples == {1, 2, 3}
    assert all(type(s) is int for s in samples)


def test_integer_range_with_fractional_bounds(rng):
    n = RangeParameter("n", 0.5, 2.5, integer=True)
    assert {n.sample(rng) for _ in range(100)} == {1, 2}


@pytest.mark.parametrize(
    "make",
    [
        lambda: RangeParameter("x", 5, -5),
        lambda: RangeParameter("x", 0.2, 0.8, integer=True),
        lambda: RangeParameter("x", 0, 1, step=0.1),
    ],
)
def test_invalid_range_rejected(make):
    with pytest.raises(ValidationError):
        make()


def test_range_requires_both_bounds():
    with pytest.raises(TypeError):
        RangeParameter("x", 0)


def test_fixed_always_returns_its_value(rng):
    assert FixedParameter("mode", "fast").sample(rng) == "fast"


def test_choice_samples_from_choices(rng):
    color = ChoiceParameter("color", ["red", "green", "blue"])
    assert {color.sample(rng) for _ in range(100)} == {"red", "green", "blue"}


def test_choice_keeps_value_types(rng):
    size = ChoiceParameter("size", [1, 2.5, "big"])
    assert {type(size.sample(rng)) for _ in range(100)} == {int, float, str}


def test_choose_n_returns_distinct_list(rng):
    picks = ChoiceParameter("tags", ["a", "b", "c", "d"], choose_n=3).sample(rng)
    assert isinstance(picks, list)
    assert len(picks) == len(set(picks)) == 3


@pytest.mark.parametrize(
    "make",
    [
        lambda: ChoiceParameter("c", []),
        lambda: ChoiceParameter("c", ["a", "b"], choose_n=3),
        lambda: ChoiceParameter("c", ["a"], choose_n=0),
    ],
)
def test_invalid_choice_rejected(make):
    with pytest.raises(ValidationError):
        make()


def test_parameters_are_immutable():
    with pytest.raises(ValidationError):
        RangeParameter("x", 0, 1).maximum = 2


def test_search_space_samples_every_parameter(rng):
    space = SearchSpace([RangeParameter("x", 0, 1), FixedParameter("f", 3), ChoiceParameter("c", ["a"])])
    assert space.names == ["x", "f", "c"]
    sample = space.sample(rng)
    assert set(sample) == {"x", "f", "c"}
    assert sample["f"] == 3
    assert sample["c"] == "a"


def test_search_space_rejects_duplicate_names():
    with pytest.raises(ValidationError, match="repeated: x"):
        SearchSpace([RangeParameter("x", 0, 1), FixedParameter("x", 2)])


def test_search_space_rejects_empty():
    with pytest.raises(ValidationError):
        SearchSpace([])


def test_search_space_from_dict():
    """A search space can be loaded from plain data, as from a YAML or JSON file."""
    space = SearchSpace.model_validate(
        {
            "parameters": [
                {"kind": "range", "name": "x", "minimum": -1, "maximum": 1},
                {"kind": "fixed", "name": "f", "value": "on"},
                {"kind": "choice", "name": "c", "choices": [1, 2]},
            ]
        }
    )
    assert [type(p) for p in space.parameters] == [RangeParameter, FixedParameter, ChoiceParameter]


def test_search_space_from_dict_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        SearchSpace.model_validate({"parameters": [{"kind": "normal", "name": "x"}]})
