import pytest

import stablebear as sb


def test_limit_gpus_never_raises_attribute_error():
    """Regression for #197: on CPU-only builds limit_gpus was not bound at
    all, so the documented no-op was actually an AttributeError."""
    try:
        sb.system.limit_gpus(1)
    except RuntimeError:
        # A CUDA build may legitimately complain when fewer GPUs are
        # available than requested; the point is the call must resolve.
        pass


def test_limit_cpus_roundtrip():
    """limit_cpus is bound on every build and must accept a positive count."""
    import os

    sb.system.limit_cpus(2)
    sb.system.limit_cpus(os.cpu_count() or 1)  # restore parallelism
