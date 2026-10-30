import os
import subprocess
import sys

import pytest

# Runs in a fresh process because CUDA reads CUDA_VISIBLE_DEVICES when it
# initializes
_SUBPROC = r"""
import numpy as np
import stablebear as sb
from stablebear import _sb_cpp as cpp

print(sb.system.build_type())
print(cpp.get_ngpus())

sb.system.set_cuda_threshold(0)
X = sb.zeros((2,), dtype=sb.pcf64)
X[0] = sb.Pcf(np.array([[0.0, 4.0],
                        [1.0, 0.0]]))
X[1] = sb.Pcf(np.array([[0.0, 1.0],
                        [1.0, 0.0]]))
print(sb.pdist(X, p=1)[0, 1])
print(np.asarray(sb.cdist(X, X, p=1))[0, 1])
print(sb.l2_kernel(X)[0, 1])
"""


def _run_without_devices():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _SUBPROC],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def test_cuda_build_without_usable_devices_falls_back_to_cpu():
    build_type, ngpus, pdist, cdist, l2_kernel = _run_without_devices()
    if build_type != "CUDA":
        pytest.skip("Requires the CUDA backend to load")

    assert ngpus == "0"
    # ||f - g||_1 = |4 - 1| * 1 = 3, and <f, g> = 4 * 1 * 1 = 4
    assert float(pdist) == 3.0
    assert float(cdist) == 3.0
    assert float(l2_kernel) == 4.0
