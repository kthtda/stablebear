"""Test that the GIL is released during blocking C++ computations.

Uses a GatedTask: a C++ StoppableTask that blocks until Python calls
advance() a specified number of times.  If wait_for() held the GIL,
the thread calling wait_for could never yield to let the main thread
call advance(), and the test would deadlock.
"""

import threading

import stablebear._sb_cpp as cpp


def test_gil_released_during_wait_for():
    """Main thread must be able to call advance() while another thread waits."""
    n_steps = 5
    task = cpp._create_gated_task(n_steps)

    # Call wait_for() in a background thread.  It blocks until the task
    # completes, releasing the GIL on each call so this (main) thread
    # can call advance().
    def wait_loop():
        while not task.wait_for(50):
            pass

    wait_thread = threading.Thread(target=wait_loop)
    wait_thread.start()

    # If the GIL is properly released during wait_for(), these advance()
    # calls will execute between wait_for() iterations and eventually
    # unblock the task.  If the GIL is NOT released, this thread is
    # starved and the join below times out.
    for _ in range(n_steps):
        task.advance()

    wait_thread.join(timeout=5.0)
    assert not wait_thread.is_alive(), (
        "wait_for() did not release the GIL -- background thread is stuck"
    )


def test_gated_task_stops_on_request():
    """request_stop() should unblock the task even without enough advances."""
    task = cpp._create_gated_task(1000)

    def wait_loop():
        while not task.wait_for(50):
            pass

    wait_thread = threading.Thread(target=wait_loop)
    wait_thread.start()

    # Don't advance -- just request stop
    task.request_stop()

    wait_thread.join(timeout=5.0)
    assert not wait_thread.is_alive(), (
        "request_stop() did not unblock the gated task"
    )
