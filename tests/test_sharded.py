"""A benchmark shared among machines: shards that together make every run, and merging them checked against the spec."""
import json

import pytest

from hone.benchmarks import sharded
from hone.benchmarks.runner import Run, random_search, run_benchmark
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
    assert {_key(r): r.without_timing() for r in whole} == {_key(r): r.without_timing() for shard in shards for r in shard}
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


def test_an_old_run_without_timing_loads_with_none(tmp_path):
    old = {"problem": "p", "optimizer": "o", "budget_per_dimension": 1, "seed": 0, "error": 0.5, "evaluations": 3, "trace": [[3, 0.5]]}
    path = tmp_path / "old.json"
    path.write_text(json.dumps([old]))
    (run,) = sharded._load_runs(path)
    assert (run.seconds, run.seconds_at, run.machine) == (None, None, None)


def test_timing_round_trips_through_saved_runs(tmp_path):
    run = Run(problem="p", optimizer="o", budget_per_dimension=1, seed=0, error=0.5, evaluations=3, trace=[(3, 0.5)],
              seconds=1.5, seconds_at={300: 0.7}, machine="CPU")
    sharded._save_runs([run], tmp_path / "r.json")
    assert sharded._load_runs(tmp_path / "r.json") == [run]


def test_a_run_records_its_time_and_machine_without_changing_its_result():
    args = dict(problems=quick_problems()[:1], optimizers={"random": random_search}, seeds=range(2), budgets_per_dimension=[10])
    plain = run_benchmark(**args)
    timed = run_benchmark(**args, time_at=(20,))
    assert [r.without_timing() for r in plain] == [r.without_timing() for r in timed]
    for p, t in zip(plain, timed):
        assert p.seconds_at is None
        assert t.seconds is not None and t.seconds > 0
        assert t.seconds_at is not None and set(t.seconds_at) == {20} and 0 < t.seconds_at[20] <= t.seconds
        assert isinstance(t.machine, str) and t.machine


def test_timing_reaches_forked_workers():
    args = dict(problems=quick_problems()[:2], optimizers={"random": random_search}, seeds=range(2), budgets_per_dimension=[10], time_at=(20,))
    runs = run_benchmark(**args, jobs=2)
    assert all(r.seconds and r.seconds_at and 20 in r.seconds_at for r in runs)
