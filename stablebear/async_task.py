#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


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
        task = task_fn()
        _wait_for_task(task, verbose=verbose)
    finally:
        if task is not None:
            task.request_stop()
            _wait_for_task(task, verbose=verbose)
