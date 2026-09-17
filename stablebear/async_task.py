from tqdm import tqdm


def _wait_for_task(task, verbose=True):
    def init_progress(task):
        progress = tqdm(
            total=task.work_total(),
            unit_scale=True,
            unit=task.work_step_unit(),
            desc=task.work_step_desc(),
        )
        return progress

    if verbose:
        progress = init_progress(task)
        work_step = task.work_step()

    wait_time_ms = 50
    while not task.wait_for(wait_time_ms):
        if verbose:
            progress.update(task.work_completed() - progress.n)
            new_work_step = task.work_step()
            if new_work_step != work_step:
                work_step = new_work_step
                print("")
                progress = init_progress(task)

    if verbose:
        progress.update(task.work_completed() - progress.n)


def _run_task(task_fn, verbose=True):
    task = None
    try:
        # _wait_for_task() polls task.wait_for(), which rethrows any exception a
        # worker stored in the task (e.g. a malformed point cloud rejected inside
        # the parallel walk) once the task completes -- so worker errors surface
        # here instead of silently returning an empty/garbage result.
        task = task_fn()
        _wait_for_task(task, verbose=verbose)
    finally:
        if task is not None:
            task.request_stop()
            try:
                # Cleanup drain: make sure the workers have actually stopped
                # before the task is destroyed. Don't let an error surfaced here
                # mask the exception (or KeyboardInterrupt) that triggered it.
                _wait_for_task(task, verbose=verbose)
            except BaseException:
                pass
