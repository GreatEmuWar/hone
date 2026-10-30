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


# --- mutation ---------------------------------------------------------------


def test_range_mutation_is_a_small_step(rng):
    x = RangeParameter("x", -50, 50)
    steps = [x.mutate(0.0, rng, scale=0.01) for _ in range(500)]
    assert all(-5 < s < 5 for s in steps)
    assert 0.5 < float(np.std(steps)) < 1.5


def test_range_mutation_stays_within_bounds(rng):
    x = RangeParameter("x", 0, 1)
    assert all(0 <= x.mutate(0.99, rng, scale=0.5) <= 1 for _ in range(200))


def test_integer_range_mutation_moves_by_whole_steps(rng):
    n = RangeParameter("n", 0, 3, integer=True)
    results = [n.mutate(1, rng, scale=0.01) for _ in range(200)]
    assert all(type(r) is int and 0 <= r <= 3 for r in results)
    assert any(r != 1 for r in results), "a tiny scale should still be able to move an integer"


def test_choice_mutation_switches_to_another_choice(rng):
    color = ChoiceParameter("color", ["red", "green", "blue"])
    results = {color.mutate("red", rng, scale=0.1) for _ in range(100)}
    assert results == {"green", "blue"}


def test_single_choice_mutation_keeps_it(rng):
    assert ChoiceParameter("c", ["only"]).mutate("only", rng, scale=0.1) == "only"


def test_choose_n_mutation_redraws_distinct_list(rng):
    tags = ChoiceParameter("tags", ["a", "b", "c"], choose_n=2)
    picks = tags.mutate(["a", "b"], rng, scale=0.1)
    assert len(picks) == len(set(picks)) == 2


def test_fixed_mutation_keeps_value(rng):
    assert FixedParameter("mode", "fast").mutate("fast", rng, scale=0.1) == "fast"


def test_search_space_lookup_by_name():
    x = RangeParameter("x", 0, 1)
    space = SearchSpace([x, FixedParameter("f", 3)])
    assert space["x"] is x
    with pytest.raises(KeyError):
        space["missing"]
