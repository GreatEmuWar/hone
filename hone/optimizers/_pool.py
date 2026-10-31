"""Worker processes that evaluate candidates, with an optional time limit on each evaluation.

concurrent.futures can't stop an evaluation once it has started, so a hung simulation would
hold up the run forever. Here each worker is a process of its own: one that runs past the
time limit, or dies, is replaced, and its evaluation counts as failed.
"""
import multiprocessing
import signal
import time
from multiprocessing.connection import Connection, wait
from typing import Any, Callable

STOP = None  # sent to a worker to make it exit
READY = "ready"  # sent by a worker once it has started and can evaluate


class _Worker:
    def __init__(self, context: Any, evaluate: Callable[[dict], float | None]) -> None:
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, evaluate), daemon=True)
        self.process.start()
        child.close()
        self.ready = False
        self.task: Any = None  # key of the candidate being evaluated, or None while idle
        self.deadline = float("inf")

    def start(self, task: Any, params: dict, timeout: float | None) -> None:
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
    # Ctrl+C reaches every process in the terminal; the main process handles it and stops the workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    connection.send(READY)
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

    def __init__(
        self,
        evaluate: Callable[[dict], float | None],
        workers: int,
        timeout: float | None = None,
        start_method: str | None = None,
    ) -> None:
        self._context = multiprocessing.get_context(start_method)
        self._evaluate = evaluate
        self._timeout = timeout
        self._workers = [self._new_worker() for _ in range(workers)]
        self._queue: list[tuple[Any, dict]] = []  # submitted, not yet started
        self._finished: list[tuple[Any, float | None]] = []

    def _new_worker(self) -> _Worker:
        return _Worker(self._context, self._evaluate)

    def map(self, candidates: list[dict]) -> list[float | None]:
        """The outcome of each candidate, in order: its value, or None if it failed, timed out or crashed its worker."""
        outcomes: list[float | None] = [None] * len(candidates)
        for index, params in enumerate(candidates):
            self.submit(index, params)
        while self.pending:
            for index, outcome in self.collect():
                outcomes[index] = outcome
        return outcomes

    def submit(self, key: Any, params: dict) -> None:
        """Queue a candidate, identified by key, to be evaluated by the next free worker."""
        self._queue.append((key, params))

    @property
    def pending(self) -> int:
        """Candidates submitted whose outcomes haven't been collected yet."""
        return len(self._queue) + sum(w.task is not None for w in self._workers) + len(self._finished)

    @property
    def idle(self) -> int:
        """Workers with nothing to do, counting those still starting up."""
        return max(sum(w.task is None for w in self._workers) - len(self._queue), 0)

    def collect(self) -> list[tuple[Any, float | None]]:
        """Wait until at least one candidate is finished, and return (key, outcome) for each one that is."""
        while not self._finished and (self._queue or any(w.task is not None for w in self._workers)):
            for worker in self._workers:
                if worker.ready and worker.task is None and self._queue:
                    key, params = self._queue.pop(0)
                    worker.start(key, params, self._timeout)
            # The time limit starts once a worker has started, so it never counts start-up time.
            watched = [w for w in self._workers if w.task is not None or not w.ready]
            deadline = min((w.deadline for w in watched), default=float("inf"))
            limit = None if deadline == float("inf") else max(deadline - time.monotonic(), 0)
            heard = wait([w.connection for w in watched], timeout=limit)
            for worker in watched:
                if worker.connection in heard:
                    self._collect(worker)
                elif time.monotonic() >= worker.deadline:
                    self._replace(worker)  # out of time: its outcome is None
        finished, self._finished = self._finished, []
        return finished

    def _collect(self, worker: _Worker) -> None:
        try:
            message = worker.connection.recv()
        except (EOFError, OSError):
            if not worker.ready:
                raise RuntimeError("a worker process failed to start; its error is printed above") from None
            self._replace(worker)  # the worker died, taking the evaluation with it
            return
        if message == READY:
            worker.ready = True
            return
        task, value, error = message
        if error is not None:
            raise error
        self._finished.append((task, value))
        worker.task = None

    def _replace(self, worker: _Worker) -> None:
        """Stop worker, counting its evaluation, if any, as failed, and start a new one in its place."""
        if worker.task is not None:
            self._finished.append((worker.task, None))
        worker.stop(force=True)
        self._workers[self._workers.index(worker)] = self._new_worker()

    def close(self, force: bool = False) -> None:
        for worker in self._workers:
            worker.stop(force)

    def __enter__(self) -> "EvaluationPool":
        return self

    def __exit__(self, kind, error, traceback) -> None:
        self.close(force=error is not None)
