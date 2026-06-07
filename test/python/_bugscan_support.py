"""Shared helpers for the bug-scan regression tests (``test_bugscan_*.py``).

These tests document *known, currently-unfixed* defects discovered by a broad
API fuzz/scan. Each test asserts the CORRECT behavior, so it FAILS today and
will pass once the underlying bug is fixed. Do **not** weaken a test to make it
green -- fix the bug instead (see ``bug-scan-findings.md`` at the repo root for
the catalogue and root-cause hints).

Several documented defects are hard crashes (SIGSEGV / SIGABRT / SIGFPE) in the
C++ backend. Running such a repro in-process would abort the entire pytest
session, so those snippets run in a fresh subprocess and we assert on the
*process outcome* instead. The subprocess uses a non-repo working directory so
it imports the *installed* ``masspcf`` (the repo root shadows the package).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap

_PREAMBLE = (
    "import numpy as np\n"
    "import masspcf as mpcf\n"
    "from masspcf import persistence, random, system\n"
    "from masspcf.point_process import poisson\n"
)


def run_snippet(body: str, timeout: float = 120):
    """Run ``body`` in a fresh Python process and return the CompletedProcess.

    A standard preamble (numpy + masspcf imports) is prepended. The working
    directory is a temp dir so the *installed* ``masspcf`` is imported rather
    than the shadowing source tree.
    """
    code = _PREAMBLE + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=tempfile.gettempdir(),
    )


def _crashed(cp) -> bool:
    # On POSIX a process killed by a signal (SIGSEGV/SIGABRT/SIGFPE) reports a
    # negative returncode; a clean uncaught Python exception exits with code 1.
    return cp.returncode < 0


def _fmt(cp) -> str:
    return (
        f"returncode={cp.returncode}\n"
        f"--- stdout ---\n{cp.stdout}\n"
        f"--- stderr (tail) ---\n{cp.stderr[-2000:]}"
    )


def assert_no_hard_crash(body: str, timeout: float = 120):
    """Assert ``body`` does not hard-crash the interpreter (no segfault/abort).

    A clean Python exception is acceptable; a signal death is not. Use this for
    crash bugs whose only firm contract is "a library must never segfault".
    """
    cp = run_snippet(body, timeout)
    assert not _crashed(cp), f"interpreter hard-crashed:\n{_fmt(cp)}"
    return cp


def assert_ok(body: str, timeout: float = 120):
    """Assert ``body`` runs to completion and prints the ``BUGSCAN_OK`` sentinel.

    ``body`` should perform its own assertions and ``print('BUGSCAN_OK')`` at
    the very end. Fails on crash, on any raised exception, or if the sentinel
    is missing. Use this for crash bugs whose correct behavior is to *succeed*
    (e.g. produce a value matching a known-good reference).
    """
    cp = run_snippet(body, timeout)
    assert not _crashed(cp), f"interpreter hard-crashed:\n{_fmt(cp)}"
    assert cp.returncode == 0, f"snippet exited non-zero:\n{_fmt(cp)}"
    assert "BUGSCAN_OK" in cp.stdout, f"missing BUGSCAN_OK sentinel:\n{_fmt(cp)}"
    return cp


def assert_clean_raises(body: str, exc=("ValueError", "TypeError"),
                        timeout: float = 120):
    """Assert ``body`` raises one of ``exc`` *cleanly* (no crash, correct type).

    ``body`` is the operation under test; it is wrapped in try/except inside the
    subprocess. ``exc`` is a tuple of exception class *names* (strings). Use
    this for crash bugs (or wrong-exception bugs) whose correct behavior is to
    raise a specific, clean Python exception.
    """
    if isinstance(exc, str):
        exc = (exc,)
    names = ", ".join(exc)
    wrapped = (
        "try:\n"
        + textwrap.indent(textwrap.dedent(body), "    ")
        + f"\nexcept ({names}) as _e:\n"
        "    print('BUGSCAN_RAISED:' + type(_e).__name__)\n"
        "except BaseException as _e:\n"
        "    print('BUGSCAN_OTHER:' + type(_e).__name__)\n"
        "    raise\n"
        "else:\n"
        "    print('BUGSCAN_NORAISE')\n"
    )
    cp = run_snippet(wrapped, timeout)
    assert not _crashed(cp), (
        f"interpreter hard-crashed instead of raising {names}:\n{_fmt(cp)}")
    assert "BUGSCAN_RAISED:" in cp.stdout, (
        f"expected a clean {names}; got:\n{_fmt(cp)}")
    return cp
