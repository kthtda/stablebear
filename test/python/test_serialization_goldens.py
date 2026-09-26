import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import stablebear as sb


GOLDEN_ROOT = Path(__file__).resolve().parents[1] / "golden" / "serialization"


def _artifacts():
    artifacts = []
    for manifest_path in sorted(GOLDEN_ROOT.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        for artifact in manifest["artifacts"]:
            artifacts.append(
                pytest.param(
                    manifest_path.parent,
                    artifact,
                    id=f"{manifest_path.parent.name}-{artifact['name']}-{artifact['serialization']}",
                )
            )
    if not artifacts:
        artifacts.append(
            pytest.param(
                None,
                None,
                marks=pytest.mark.skip(reason="no serialization golden files committed"),
                id="no-goldens",
            )
        )
    return artifacts


def _assert_array(actual, expected):
    expected_array = np.asarray(expected["values"], dtype=expected["dtype"])
    expected_array = expected_array.reshape(expected["shape"])
    actual_array = np.asarray(actual)
    assert actual_array.dtype == expected_array.dtype
    assert actual_array.shape == expected_array.shape
    assert np.array_equal(actual_array, expected_array, equal_nan=True)


def _assert_expected(actual, expected):
    assert type(actual).__name__ in expected["accepted_types"]
    comparison = expected["comparison"]
    if comparison == "singleton":
        assert actual is getattr(sb, expected["name"])
        return

    if "dtype" in expected:
        dtype = getattr(actual, "dtype", None)
        dtype_name = getattr(dtype, "name", getattr(dtype, "__name__", str(dtype)))
        assert dtype_name == expected["dtype"]
    if "shape" in expected:
        assert tuple(actual.shape) == tuple(expected["shape"])

    if comparison == "array":
        _assert_array(actual, expected["array"])
    elif comparison == "elements":
        for element in expected["elements"]:
            _assert_array(actual[tuple(element["index"])], element["array"])
    else:
        raise AssertionError(f"Unknown golden comparison {comparison!r}")


@pytest.mark.parametrize("directory,artifact", _artifacts())
def test_serialization_golden(directory, artifact):
    path = directory / artifact["file"]
    data = path.read_bytes()
    assert len(data) == artifact["bytes"]
    assert hashlib.sha256(data).hexdigest() == artifact["sha256"]

    if artifact["serialization"] == "pickle":
        actual = pickle.loads(data)
    elif artifact["serialization"] == "binary":
        with path.open("rb") as file:
            actual = sb.load(file)
    else:
        raise AssertionError(
            f"Unknown golden serialization {artifact['serialization']!r}"
        )
    _assert_expected(actual, artifact["expected"])
