#!/usr/bin/env python3
"""Generate V1 binary fixtures with masspcf 0.4.0b8."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import masspcf
import numpy as np


EXPECTED_VERSION = "0.4.0b8"


def _expected(array, legacy_type):
    return {
        "accepted_types": ["FloatTensor", legacy_type],
        "comparison": "array",
        "array": {
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "values": array.tolist(),
        },
        "dtype": array.dtype.name,
        "shape": list(array.shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="serialization golden root (default: test/golden/serialization)",
    )
    args = parser.parse_args()

    version = importlib.metadata.version("masspcf")
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"Expected masspcf {EXPECTED_VERSION}, but found {version}"
        )

    output = args.output_root / version
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cases = (
        ("tensor_float32", masspcf.Float32Tensor, np.float32),
        ("tensor_float64", masspcf.Float64Tensor, np.float64),
    )
    artifacts = []
    for name, tensor_type, dtype in cases:
        array = np.asarray([1.25, -2.5, 4.0], dtype=dtype)
        tensor = tensor_type(array)
        filename = name + ".sb"
        path = output / filename
        with path.open("wb") as file:
            masspcf.save(tensor, file)
        data = path.read_bytes()
        artifacts.append({
            "name": name,
            "description": f"V1 contiguous {array.dtype.name} tensor",
            "serialization": "binary",
            "file": filename,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "expected": _expected(array, tensor_type.__name__),
        })

    backend = type(cases[0][1](np.asarray([0], dtype=np.float32))._data).__module__
    manifest = {
        "schema_version": 1,
        "producer": {
            "package": "masspcf",
            "version": version,
            "label": version,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "backend": backend,
        },
        "artifacts": artifacts,
        "unsupported": [],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(artifacts)} V1 binary artifacts in {output}")


if __name__ == "__main__":
    main()
