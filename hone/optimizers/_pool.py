"""Worker processes that evaluate candidates, with an optional time limit on each evaluation.

concurrent.futures can't stop an evaluation once it has started, so a hung simulation would
hold up the run forever. Here each worker is a process of its own: one that runs past the
time limit, or dies, is replaced, and its evaluation counts as failed.
"""
import multiprocessing
import time
from multiprocessing.connection import Connection, wait
from typing import Any, Callable

STOP = None  # sent to a worker to make it exit


class _Worker:
    def __init__(self, context: Any, evaluate: Callable[[dict], float | None]) -> None:
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, evaluate), daemon=True)
        self.process.start()
        child.close()
        self.task: int | None = None  # index of the candidate being evaluated
        self.deadline = float("inf")

    def start(self, task: int, params: dict, timeout: float | None) -> None:
        self.connection.send((task, params))
        self.task = task
        self.deadline = time.monotonic() + timeout if timeout is not None else float("inf")

    def stop(self, force: bool = False) -> None:
        if not force and self.process.is_alive():
            try:
                self.connection.send(STOP)
                self.process.join(timeout=1)
            except (BrokenPipeError, OSError):
                pass
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.connection.close()


def _serve(connection: Connection, evaluate: Callable[[dict], float | None]) -> None:
    """A worker's loop: evaluate each candidate it is sent, and send back the outcome."""
    while (message := connection.recv()) is not STOP:
        task, params = message
        try:
            connection.send((task, evaluate(params), None))
        except Exception as error:
            try:
                connection.send((task, None, error))
            except Exception:  # the error itself can't be pickled
                connection.send((task, None, RuntimeError(f"{type(error).__name__}: {error}")))


class EvaluationPool:
    """Evaluates candidates in worker processes, each within an optional time limit in seconds.

    evaluate(params) runs in the workers; it returns the value, or None for a failed
    evaluation, and any exception it raises stops every worker and is raised again here.
    """

    def __init__(self, evaluate: Callable[[dict], float | None], workers: int, timeout: float | None = None) -> None:
        self._context = multiprocessing.get_context()
        self._evaluate = evaluate
        self._timeout = timeout
        self._workers = [self._new_worker() for _ in range(workers)]

    def _new_worker(self) -> _Worker:
        return _Worker(self._context, self._evaluate)

    def map(self, candidates: list[dict]) -> list[float | None]:
        """The outcome of each candidate, in order: its value, or None if it failed, timed out or crashed its worker."""
        outcomes: list[float | None] = [None] * len(candidates)
        waiting = list(range(len(candidates)))[::-1]
        while waiting or any(w.task is not None for w in self._workers):
            for worker in self._workers:
                if worker.task is None and waiting:
                    task = waiting.pop()
                    worker.start(task, candidates[task], self._timeout)
            busy = [w for w in self._workers if w.task is not None]
            deadline = min(w.deadline for w in busy)
            limit = None if deadline == float("inf") else max(deadline - time.monotonic(), 0)
            ready = wait([w.connection for w in busy], timeout=limit)
            for worker in busy:
                if worker.connection in ready:
                    self._collect(worker, outcomes)
                elif time.monotonic() >= worker.deadline:
                    self._replace(worker)  # out of time: its outcome stays None
        return outcomes

    def _collect(self, worker: _Worker, outcomes: list[float | None]) -> None:
        try:
            task, value, error = worker.connection.recv()
        except (EOFError, OSError):
            self._replace(worker)  # the worker died, taking the evaluation with it
            return
        if error is not None:
            raise error
        outcomes[task] = value
        worker.task = None

    def _replace(self, worker: _Worker) -> None:
        worker.stop(force=True)
        self._workers[self._workers.index(worker)] = self._new_worker()

    def close(self, force: bool = False) -> None:
        for worker in self._workers:
            worker.stop(force)

    def __enter__(self) -> "EvaluationPool":
        return self

    def __exit__(self, kind, error, traceback) -> None:
        self.close(force=error is not None)
