"""A pool for forkserver workers has the forkserver import hone once, so a replaced worker doesn't import it again."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hone.optimizers import _pool
from hone.optimizers._pool import EvaluationPool


def evaluate(params):
    return 0.0


@pytest.mark.parametrize("method, preloads", [("forkserver", True), ("fork", False), ("spawn", False)])
def test_only_a_forkserver_is_told_to_preload_hone(monkeypatch, method, preloads):
    context = MagicMock()
    context.get_start_method.return_value = method
    monkeypatch.setattr(_pool, "multiprocessing", SimpleNamespace(get_context=lambda start_method=None: context))
    EvaluationPool(evaluate, workers=0, start_method=method)  # no workers, so nothing is started
    if preloads:
        context.set_forkserver_preload.assert_called_once_with(["__main__", "hone"])
    else:
        context.set_forkserver_preload.assert_not_called()
