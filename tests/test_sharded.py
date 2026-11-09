"""A benchmark shared among machines: shards that together make every run, and merging them checked against the spec."""
import json

import pytest

from hone.benchmarks import sharded
from hone.benchmarks.runner import random_search, run_benchmark
from hone.benchmarks.problems import quick_problems

SPEC = '''
from hone.benchmarks.problems import quick_problems
from hone.benchmarks.runner import random_search

def problems():
    return quick_problems()[:2]

def optimizers():
    return {"random": random_search}

SEEDS = 3
BUDGETS = (10, 20)

def report(runs):
    return f"checked {len(runs)} runs\\n"
'''


@pytest.fixture
def spec_file(tmp_path):
    path = tmp_path / "spec.py"
    path.write_text(SPEC)
    return path


def _key(run):
    return (run.problem, run.optimizer, run.budget_per_dimension, run.seed)


def test_the_shards_make_every_run_once_and_the_same_runs():
    problems, optimizers = quick_problems()[:2], {"random": random_search}
    whole = run_benchmark(problems, optimizers, range(3), (10, 20))
    shards = [run_benchmark(problems, optimizers, range(3), (10, 20), shard=(i, 5)) for i in range(5)]
    assert sorted(map(_key, whole)) == sorted(_key(run) for shard in shards for run in shard)
    assert {_key(r): r for r in whole} == {_key(r): r for shard in shards for r in shard}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_every_optimizer_runs_a_seed_in_the_same_shard():
    """So a comparison between optimizers, seed by seed, is made on one machine's maths libraries."""
    problems, optimizers = quick_problems()[:2], {"random": random_search, "again": random_search}
    shards = [run_benchmark(problems, optimizers, range(3), (10, 20), shard=(i, 4)) for i in range(4)]
    where = {}
    for i, shard in enumerate(shards):
        for run in shard:
            where.setdefault((run.problem, run.budget_per_dimension, run.seed), set()).add(i)
    assert len(where) == 12 and all(len(found) == 1 for found in where.values())


def test_a_shard_that_doesnt_exist_is_refused():
    with pytest.raises(ValueError, match="doesn't exist"):
        run_benchmark(quick_problems()[:1], {"random": random_search}, range(1), (10,), shard=(3, 3))


def test_merging_refuses_missing_or_repeated_runs(spec_file):
    spec = sharded.load_spec(spec_file)
    shards = [sharded.run_shard(spec, i, 3) for i in range(3)]
    assert len(sharded.merge(spec, shards)) == len(sharded.expected_runs(spec)) == 12
    with pytest.raises(ValueError, match="4 runs missing"):
        sharded.merge(spec, shards[:2])
    with pytest.raises(ValueError, match="4 repeated"):
        sharded.merge(spec, [*shards, shards[0]])


def test_a_spec_must_define_what_to_run(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text("SEEDS = 3\n")
    with pytest.raises(ValueError, match="doesn't define problems, optimizers, BUDGETS"):
        sharded.load_spec(path)


def test_the_command_line_runs_shards_and_merges_them(spec_file, tmp_path, capsys):
    outs = [tmp_path / f"runs-{i}.json" for i in range(2)]
    for i, out in enumerate(outs):
        assert sharded.main(["run", str(spec_file), "--shard", str(i), "--of", "2", "--out", str(out)]) == 0
    markdown = tmp_path / "results.md"
    argv = ["merge", str(spec_file), *map(str, outs), "--runs", str(tmp_path / "all.json"), "--markdown", str(markdown)]
    assert sharded.main(argv) == 0
    assert len(json.loads((tmp_path / "all.json").read_text())) == 12
    assert "checked 12 runs" in markdown.read_text() and "in 2 shards" in markdown.read_text()
    assert sharded.main(["merge", str(spec_file), str(outs[0])]) == 1
    assert "runs missing" in capsys.readouterr().err


def test_the_smoke_spec_loads():
    spec = sharded.load_spec("benchmarks/specs/smoke.py")
    assert len(sharded.expected_runs(spec)) == 6 * 2 * 4


def test_a_factory_can_skip_some_seeds_of_an_expensive_cell():
    from hone.benchmarks.runner import SkipBudget

    def few(problem, budget, seed):
        if budget > 50 and seed >= 1:
            raise SkipBudget("one seed is enough here")
        return random_search(problem, budget, seed)

    runs = run_benchmark(quick_problems()[:1], {"few": few}, range(3), (10, 20))
    assert sorted((r.budget_per_dimension, r.seed) for r in runs) == [(10, 0), (10, 1), (10, 2), (20, 0)]
