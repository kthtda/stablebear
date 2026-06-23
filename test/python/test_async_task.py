"""Control-flow tests for _run_task's exception handling.

These use a fake task that drives wait_for() deterministically, so we can
exercise the interrupt/cleanup path without having to inject a real
KeyboardInterrupt into a blocking C++ wait. The C++ side (wait_for actually
rethrowing a worker's exception once the task completes) is covered end-to-end
by test_bugscan_persistence.py.
"""

import pytest

from stablebear.async_task import _run_task


class _FakeTask:
    """Mimics the subset of the StoppableTask API that _run_task uses.

    wait_for() returns/raises according to `on_poll`, which receives whether
    request_stop() has been called yet so a test can behave differently in the
    main poll loop vs. the post-stop cleanup drain.
    """

    def __init__(self, on_poll):
        self._on_poll = on_poll
        self.stopped = False
        self.poll_count = 0

    def request_stop(self):
        self.stopped = True

    def wait_for(self, _ms):
        self.poll_count += 1
        return self._on_poll(self)


def test_interrupt_swallows_drain_error_and_preserves_interrupt():
    """If the poll loop is interrupted while a worker error is still pending,
    the cleanup drain's error is swallowed and the interrupt propagates."""

    def on_poll(task):
        if not task.stopped:
            # Ctrl-C arriving while blocked in the main wait_for() loop.
            raise KeyboardInterrupt
        # After request_stop(), the drain observes the worker's error -- which
        # must not mask the KeyboardInterrupt that triggered cleanup.
        raise RuntimeError("worker error surfaced during drain")

    task = _FakeTask(on_poll)

    with pytest.raises(KeyboardInterrupt):
        _run_task(lambda: task, verbose=False)

    assert task.stopped, "request_stop() should run in the finally block"
    assert task.poll_count >= 2, "cleanup drain should poll after the interrupt"


def test_worker_error_propagates_from_poll_loop():
    """A worker error surfaced by wait_for() during the normal loop propagates
    out of _run_task (the drain doesn't suppress the original failure)."""

    def on_poll(_task):
        raise RuntimeError("worker blew up")

    task = _FakeTask(on_poll)

    with pytest.raises(RuntimeError, match="worker blew up"):
        _run_task(lambda: task, verbose=False)

    assert task.stopped


def test_clean_run_completes_without_error():
    """A task that finishes normally returns without raising; the cleanup drain
    sees an already-finished task and stays quiet."""

    def on_poll(task):
        # Finish immediately, both in the main loop and the drain.
        return True

    task = _FakeTask(on_poll)

    _run_task(lambda: task, verbose=False)

    assert task.stopped
