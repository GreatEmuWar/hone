"""Conditional parameters: writing conditions with >>, points without the parameters they switch off, and runs over them."""
import json

import numpy as np
import pytest
from pydantic import ValidationError

from hone import (
    CMAES,
    BayesianConfig,
    BayesianOptimizer,
    ChoiceParameter,
    CMAESConfig,
    Condition,
    Constraint,
    FixedParameter,
    GeneticOptimizer,
    GeneticOptimizerConfig,
    OptimizerConfig,
    RandomSearch,
    RangeParameter,
    Result,
    SearchSpace,
)
from hone.optimizers.genetic.variation import mutate

MODEL = ChoiceParameter("model", ["svm", "forest"])
KERNEL = ChoiceParameter("kernel", ["linear", "rbf"])
C = RangeParameter("C", 0.01, 100, log=True)
GAMMA = RangeParameter("gamma", 1e-4, 1, log=True)
TREES = RangeParameter("trees", 10, 500, integer=True)
LAYERS = RangeParameter("layers", 1, 4, integer=True)
UNITS = RangeParameter("units_3", 8, 64, integer=True)

SPACE = SearchSpace(
    [MODEL, KERNEL, C, GAMMA, TREES],
    conditions=[MODEL["svm"] >> [KERNEL, C], KERNEL["rbf"] >> GAMMA, MODEL["forest"] >> TREES],
)


def score(model, C=None, kernel=None, gamma=None, trees=None):
    """Lowest for an svm with an rbf kernel, C near 1 and gamma near 0.01; a forest is never as good."""
    if model == "forest":
        return 1 + abs(np.log10(trees) - 2)
    error = abs(np.log10(C))
    return error + (abs(np.log10(gamma) + 2) if kernel == "rbf" else 0.5)


def random_search(space, **config):
    return RandomSearch(space, OptimizerConfig(**{"max_evaluations": 200, "seed": 0, **config}))


# --- writing conditions --------------------------------------------------------


def test_a_choices_values_on_the_left_of_shift_make_a_condition():
    condition = MODEL["svm"] >> C
    assert isinstance(condition, Condition)
    assert condition.parameters == ["C"] and str(condition) == "model['svm'] >> C"
    assert str(MODEL["svm", "forest"] >> [KERNEL, C]) == "model['svm', 'forest'] >> [kernel, C]"


def test_a_comparison_in_brackets_makes_a_condition_on_a_range():
    assert str((LAYERS >= 3) >> UNITS) == "(layers >= 3) >> units_3"


def test_a_comparison_without_brackets_explains_the_fix():
    with pytest.raises(TypeError, match=r"put the comparison in brackets, such as `\(layers >= 3\) >> units_3`"):
        LAYERS >= 3 >> UNITS  # type: ignore[operator]  # noqa: B015


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: MODEL >> C, r"pick the values that do, such as model\['svm'\]"),
        (lambda: LAYERS >> UNITS, r"compare it in brackets"),
        (lambda: MODEL["svn"], r"'svn' isn't one of model's choices"),
        (lambda: MODEL["svm"] >> 3, r"the right of >> is a parameter"),
        (lambda: MODEL["svm"] >> [], r"switches nothing on"),
        (lambda: ChoiceParameter("tags", ["a", "b", "c"], choose_n=2)["a"], r"picks 2 choices at once"),
        (lambda: (C + GAMMA) >> TREES, r"the left of `>> trees` must be a test"),
    ],
)
def test_mistakes_with_shift_explain_themselves(build, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build()


def test_a_test_has_no_truth_value():
    with pytest.raises(TypeError, match="for the left of >>"):
        bool(MODEL["svm"])


def test_conditions_and_constraints_go_in_their_own_lists():
    with pytest.raises(TypeError, match="goes in conditions="):
        SearchSpace([MODEL, C], [MODEL["svm"] >> C])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="goes in constraints="):
        SearchSpace([MODEL, C], conditions=[MODEL["svm"] >> (C <= 5)])  # type: ignore[list-item]


# --- validating a search space -------------------------------------------------


@pytest.mark.parametrize(
    "parameters, conditions, message",
    [
        ([C], [MODEL["svm"] >> C], "tests 'model', which isn't a parameter"),
        ([MODEL], [MODEL["svm"] >> C], "switches on 'C', which isn't a parameter"),
        ([MODEL, C], [MODEL["svm"] >> C, MODEL["forest"] >> C], r"write `model\['svm', 'forest'\] >> C`"),
        ([MODEL, KERNEL, C], [MODEL["svm"] >> C, KERNEL["rbf"] >> C], "can have only one"),
        ([MODEL, C], [MODEL["svm"] >> MODEL], "depend on itself"),
    ],
)
def test_conditions_are_checked_against_the_search_space(parameters, conditions, message):
    with pytest.raises(ValidationError, match=message):
        SearchSpace(parameters, conditions=conditions)


def test_a_condition_tests_a_choice_parameter_by_its_values():
    renamed = ChoiceParameter("model", ["svm", "forest", "knn"])
    with pytest.raises(ValidationError, match="'knn', which isn't one of model's choices"):
        SearchSpace([MODEL, C], conditions=[renamed["knn"] >> C])
    with pytest.raises(ValidationError, match="needs a choice parameter"):
        SearchSpace([RangeParameter("model", 0, 1), C], conditions=[MODEL["svm"] >> C])


def test_a_loop_of_conditions_is_refused():
    with pytest.raises(ValidationError, match="the conditions form a loop: C needs gamma needs C"):
        SearchSpace([C, GAMMA], conditions=[(GAMMA >= 0.1) >> C, (C >= 1) >> GAMMA])


def test_a_search_space_with_conditions_saves_and_loads():
    space = SearchSpace(
        [MODEL, KERNEL, C, GAMMA, TREES, LAYERS, UNITS],
        constraints=[MODEL["svm"] >> (C * GAMMA <= 10)],
        conditions=[*SPACE.conditions, (LAYERS >= 3) >> UNITS],
    )
    loaded = SearchSpace.model_validate_json(space.model_dump_json())
    assert loaded == space
    assert loaded.active({"model": "forest", "trees": 20, "layers": 3, "units_3": 9, "C": 1}) == {
        "model": "forest",
        "trees": 20,
        "layers": 3,
        "units_3": 9,
    }


# --- the point an objective sees -----------------------------------------------


def test_active_leaves_out_what_the_conditions_switch_off():
    everything = {"model": "svm", "kernel": "linear", "C": 1.0, "gamma": 0.1, "trees": 50}
    assert SPACE.active(everything) == {"model": "svm", "kernel": "linear", "C": 1.0}
    assert SPACE.active({**everything, "kernel": "rbf"}) == {"model": "svm", "kernel": "rbf", "C": 1.0, "gamma": 0.1}
    # gamma depends on kernel, which a forest switches off, so gamma is off too.
    assert SPACE.active({**everything, "model": "forest", "kernel": "rbf"}) == {"model": "forest", "trees": 50}


def test_parameters_can_come_before_the_ones_they_depend_on():
    space = SearchSpace([GAMMA, KERNEL], conditions=[KERNEL["rbf"] >> GAMMA])
    assert space.active({"gamma": 0.1, "kernel": "rbf"}) == {"gamma": 0.1, "kernel": "rbf"}
    assert space.check({"kernel": "linear"}) == {"kernel": "linear"}


def test_a_point_gives_values_only_for_the_parameters_switched_on():
    assert SPACE.check({"model": "forest", "trees": 20}) == {"model": "forest", "trees": 20}
    with pytest.raises(ValueError, match="C isn't used at this point, since it's only used when model is 'svm'"):
        SPACE.check({"model": "forest", "trees": 20, "C": 1})
    with pytest.raises(ValueError, match="missing values for: C, gamma"):
        SPACE.check({"model": "svm", "kernel": "rbf"})


def test_a_fixed_parameter_can_be_conditional():
    space = SearchSpace([MODEL, FixedParameter("probability", True)], conditions=[MODEL["svm"] >> FixedParameter("probability", True)])
    assert space.check({"model": "svm"}) == {"model": "svm", "probability": True}
    assert space.check({"model": "forest"}) == {"model": "forest"}


def test_samples_hold_every_parameter():
    rng = np.random.default_rng(0)
    points = [SPACE.sample(rng) for _ in range(50)]
    assert all(set(point) == set(SPACE.names) for point in points)
    assert {SPACE.active(point)["model"] for point in points} == {"svm", "forest"}


# --- constraints on conditional parameters -------------------------------------


def test_a_constraint_applies_only_while_its_parameters_are_switched_on():
    space = SearchSpace([MODEL, C, TREES], [C * TREES <= 100], [MODEL["svm"] >> C, MODEL["forest"] >> TREES])
    assert space.feasible({"model": "svm", "C": 50, "trees": 400})  # no trees with an svm
    space = SearchSpace([MODEL, C, TREES], [C <= 10], [MODEL["svm"] >> C])
    assert not space.feasible({"model": "svm", "C": 50}) and space.feasible({"model": "forest", "C": 50})


def test_a_constraint_after_shift_applies_only_while_the_test_holds():
    constraint = MODEL["svm"] >> (C <= 10)
    assert str(constraint) == "model['svm'] >> (C <= 10)"
    space = SearchSpace([MODEL, C], [constraint])
    assert space.feasible({"model": "forest", "C": 50}) and not space.feasible({"model": "svm", "C": 50})
    assert [str(c) for c in space.broken({"model": "svm", "C": 50})] == ["model['svm'] >> (C <= 10)"]
    nested = MODEL["svm"] >> ((C >= 1) >> (C <= 10))
    assert SearchSpace.model_validate_json(SearchSpace([MODEL, C], [nested]).model_dump_json()).constraints == [nested]


def test_a_function_constraint_is_skipped_while_a_parameter_it_needs_is_switched_off():
    small = Constraint.function(lambda C, trees: C * trees - 100, name="small")
    space = SearchSpace([MODEL, C, TREES], [small], [MODEL["svm"] >> C, MODEL["forest"] >> TREES])
    assert space.feasible({"model": "svm", "C": 50, "trees": 400})
    conditional = SearchSpace([MODEL, C, TREES], [MODEL["svm"] >> Constraint.function(lambda C: C - 10, name="c")])
    assert conditional.feasible({"model": "forest", "C": 50, "trees": 1})
    rebuilt = SearchSpace.model_validate_json(conditional.model_dump_json()).with_functions({"c": lambda C: C - 10})
    assert not rebuilt.feasible({"model": "svm", "C": 50, "trees": 1})


# --- runs ----------------------------------------------------------------------


def test_random_search_asks_for_points_without_switched_off_parameters():
    optimizer = random_search(SPACE)
    trials = optimizer.ask()
    assert all(SPACE.check(trial.params) == trial.params for trial in trials)
    assert {trial.params["model"] for trial in trials} == {"svm", "forest"}


def test_random_search_finds_the_best_branch():
    result = random_search(SPACE, max_evaluations=400).optimize(score)
    assert result.best_params["model"] == "svm" and result.best_params["kernel"] == "rbf"
    assert set(result.best_params) == {"model", "kernel", "C", "gamma"}
    assert all(set(stats.best_params or {}) <= set(SPACE.names) for stats in result.generations)
    assert SPACE.check(result.best_params) == result.best_params


def test_the_objective_needs_a_default_for_each_conditional_parameter():
    def no_defaults(model, C, kernel, gamma, trees):
        return 0.0

    with pytest.raises(ValueError, match="isn't passed C, gamma, kernel, trees at points where a condition"):
        random_search(SPACE).optimize(no_defaults)
    random_search(SPACE, max_evaluations=20).optimize(lambda model, **rest: 0.0)


def test_initial_points_leave_out_switched_off_parameters():
    optimizer = RandomSearch(SPACE, OptimizerConfig(seed=0), initial_points=[{"model": "forest", "trees": 100}])
    assert optimizer.ask()[0].params == {"model": "forest", "trees": 100}
    with pytest.raises(ValueError, match="trees isn't used at this point"):
        RandomSearch(SPACE, initial_points={"model": "svm", "kernel": "linear", "C": 1, "trees": 100})


def test_points_that_differ_only_where_switched_off_are_averaged_together():
    optimizer = random_search(SPACE, noise=True, population_size=2, max_evaluations=2)
    first, second = optimizer.ask()
    key = optimizer._key
    assert key({"model": "forest", "trees": 20, "C": 1.0}) == key({"model": "forest", "trees": 20, "C": 5.0})
    optimizer.tell(first, 1.0)
    optimizer.tell(second, 2.0)
    assert optimizer.result().best_params == SPACE.check(first.params)


def test_a_result_with_conditions_saves_rebuilds_and_repeats(tmp_path):
    result = random_search(SPACE, max_evaluations=60).optimize(score)
    loaded = Result.model_validate_json(result.model_dump_json())
    assert loaded.run is not None and loaded.run.search_space == SPACE
    assert loaded.rebuild_optimizer().optimize(score) == result
    assert json.loads(result.model_dump_json())["run"]["search_space"]["conditions"][1]["when"] == {
        "kind": "choice",
        "name": "kernel",
        "values": ["rbf"],
    }


def test_a_checkpointed_run_with_conditions_resumes(tmp_path):
    whole = random_search(SPACE, population_size=10).optimize(score)
    partial = random_search(SPACE, population_size=10)
    for _ in range(5):
        for trial in partial.ask():
            partial.tell(trial, score(**trial.params))
    partial.save_checkpoint(tmp_path / "run.checkpoint")
    resumed = random_search(SPACE, population_size=10)
    resumed.load_checkpoint(tmp_path / "run.checkpoint")
    while not resumed.done:
        for trial in resumed.ask():
            resumed.tell(trial, score(**trial.params))
    assert resumed.result() == whole


OPTIMIZERS = [
    (RandomSearch, OptimizerConfig),
    (GeneticOptimizer, GeneticOptimizerConfig),
    (BayesianOptimizer, BayesianConfig),
]


@pytest.mark.parametrize("optimizer_class, config_class", OPTIMIZERS)
def test_every_optimizer_only_passes_switched_on_parameters(optimizer_class, config_class):
    seen = []

    def recording(model, **rest):
        seen.append({"model": model, **rest})
        return score(model, **rest)

    budget = 40 if optimizer_class is BayesianOptimizer else 300
    result = optimizer_class(SPACE, config_class(max_evaluations=budget, seed=0)).optimize(recording)
    assert len(seen) == budget and all(SPACE.check(point) == point for point in seen)
    assert SPACE.check(result.best_params) == result.best_params
    assert result.best_params["model"] == "svm" and result.best_params["kernel"] == "rbf"


def test_cma_es_searches_ranges_that_conditions_switch_on():
    space = SearchSpace([LAYERS, UNITS, C], conditions=[(LAYERS >= 3) >> UNITS])

    def objective(layers, C, units_3=None):
        return (layers - 3) ** 2 + abs(np.log10(C)) + (abs(units_3 - 20) / 10 if units_3 is not None else 1)

    seen = []
    result = CMAES(space, CMAESConfig(max_evaluations=300, seed=0)).optimize(lambda **p: seen.append(p) or objective(**p))
    assert all(("units_3" in point) == (point["layers"] >= 3) for point in seen)
    assert result.best_params["layers"] == 3 and abs(result.best_params["units_3"] - 20) <= 2


def test_the_genetic_optimizer_mutates_only_genes_switched_on():
    optimizer = GeneticOptimizer(SPACE, GeneticOptimizerConfig(max_evaluations=100, seed=0, mutation_rate=1.0, crossover_rate=1.0))
    parent = {"model": "forest", "kernel": "rbf", "C": 1.0, "gamma": 0.01, "trees": 100}
    for _ in range(20):
        child = mutate(parent, SPACE, optimizer.rng, rate=1.0, scale=0.1, only=set(SPACE.active(parent)))
        assert {name: child[name] for name in ("kernel", "C", "gamma")} == {"kernel": "rbf", "C": 1.0, "gamma": 0.01}


def test_crossover_takes_a_gene_from_the_parent_it_is_switched_on_in():
    optimizer = GeneticOptimizer(SPACE, GeneticOptimizerConfig(max_evaluations=100, seed=0))
    svm = {"model": "svm", "kernel": "rbf", "C": 1.0, "gamma": 0.01, "trees": 400}
    forest = {"model": "forest", "kernel": "linear", "C": 50.0, "gamma": 0.5, "trees": 100}
    mother, father = optimizer._aligned(svm, forest)
    assert mother == {**svm, "trees": 100} and father == {**forest, "kernel": "rbf", "C": 1.0, "gamma": 0.01}


def test_the_genetic_optimizer_breeds_each_branch_in_its_own_niche():
    config = GeneticOptimizerConfig(max_evaluations=600, seed=1, niche_share=0.5)
    result = GeneticOptimizer(SPACE, config).optimize(score)
    assert result.best_fitness < 0.05


def test_bayesian_optimization_sees_points_that_differ_only_where_switched_off_as_one():
    encoding = BayesianOptimizer(SPACE, BayesianConfig(seed=0))._encoding
    forest = {"model": "forest", "kernel": "rbf", "C": 1.0, "gamma": 0.01, "trees": 100}
    assert np.array_equal(encoding.encode(forest), encoding.encode({**forest, "kernel": "linear", "C": 50.0}))
    # A candidate holding a value in a column its point switches off is snapped to the same encoding.
    linear = encoding.encode({**forest, "model": "svm", "kernel": "linear"})
    candidate = linear.copy()
    candidate[encoding.columns - 1] = 0.9  # the trees column, which an svm switches off
    assert np.array_equal(encoding.snap(candidate[None, :])[0], linear)


@pytest.mark.parametrize("optimizer_class, config_class", [*OPTIMIZERS, (CMAES, CMAESConfig)])
def test_without_conditions_every_optimizer_runs_as_before(optimizer_class, config_class):
    space = SearchSpace([C, GAMMA])

    def objective(C, gamma):
        return abs(np.log10(C)) + abs(np.log10(gamma) + 2)

    budget = 30 if optimizer_class is BayesianOptimizer else 100
    first = optimizer_class(space, config_class(max_evaluations=budget, seed=0)).optimize(objective)
    assert first == optimizer_class(space, config_class(max_evaluations=budget, seed=0)).optimize(objective)


def test_without_conditions_runs_are_unchanged():
    space = SearchSpace([C, GAMMA])
    assert space.active({"C": 1, "gamma": 2}) == {"C": 1, "gamma": 2}
    assert not space.is_conditional("C")
