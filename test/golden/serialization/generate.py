#!/usr/bin/env python3
"""Generate serialization fixtures with the installed Stablebear version.

Run this script in an environment containing the historical version to test,
then commit the resulting ``test/golden/serialization/<version>/`` directory.
The script is intentionally self-contained so it can be copied to an older
checkout.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import pickle
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stablebear as sb


PICKLE_PROTOCOL = 4
PACKAGE_NAME = "stablebear"


def _distribution_version():
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        pass
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    if pyproject.exists():
        match = re.search(
            r'^version\s*=\s*["\']([^"\']+)["\']',
            pyproject.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1)
    return "unknown"


def _dtype_name(obj):
    dtype = getattr(obj, "dtype", None)
    return getattr(dtype, "name", getattr(dtype, "__name__", str(dtype)))


def _json_array(value):
    array = np.asarray(value)
    return {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _array_expectation(obj, accepted_types=None):
    expected = {
        "accepted_types": accepted_types or [type(obj).__name__],
        "comparison": "array",
        "array": _json_array(obj),
    }
    if hasattr(obj, "dtype"):
        expected["dtype"] = _dtype_name(obj)
    if hasattr(obj, "shape"):
        expected["shape"] = list(obj.shape)
    return expected


def _element_expectation(obj, accepted_types=None):
    shape = tuple(obj.shape)
    elements = []
    for index in np.ndindex(*shape):
        elements.append({"index": list(index), "array": _json_array(obj[index])})
    return {
        "accepted_types": accepted_types or [type(obj).__name__],
        "comparison": "elements",
        "dtype": _dtype_name(obj),
        "shape": list(shape),
        "elements": elements,
    }


def _singleton_expectation(obj):
    return {
        "accepted_types": [type(obj).__name__],
        "comparison": "singleton",
        "name": getattr(obj, "name", getattr(obj, "__name__", str(obj))),
    }


@dataclass
class Case:
    name: str
    description: str
    factory: object
    expectation: object
    binary: bool = True


def _case(cases, name, description, factory, expectation, binary=True):
    cases.append(Case(name, description, factory, expectation, binary))


def _available(name):
    return hasattr(sb, name)


def _barcode_class():
    try:
        return importlib.import_module(f"{PACKAGE_NAME}.persistence").Barcode
    except (ImportError, AttributeError):
        return None


def _barcode_tensor_class():
    if hasattr(sb, "BarcodeTensor"):
        return sb.BarcodeTensor
    try:
        return importlib.import_module(
            f"{PACKAGE_NAME}.persistence"
        ).BarcodeTensor
    except (ImportError, AttributeError):
        return None


def _cases():
    cases = []

    numeric = (
        ("float32", "FloatTensor", np.float32, [1.25, -2.5, 4.0]),
        ("float64", "FloatTensor", np.float64, [1.25, -2.5, 4.0]),
        ("int32", "IntTensor", np.int32, [1, -2, 4]),
        ("int64", "IntTensor", np.int64, [1, -2, 4]),
        ("uint32", "IntTensor", np.uint32, [1, 2, 4]),
        ("uint64", "IntTensor", np.uint64, [1, 2, 4]),
        ("bool", "BoolTensor", np.bool_, [True, False, True]),
    )
    for case_name, class_name, dtype, values in numeric:
        if _available(class_name):
            _case(
                cases,
                f"tensor_{case_name}",
                f"Contiguous {case_name} tensor with shape (3,)",
                lambda class_name=class_name, dtype=dtype, values=values: getattr(
                    sb, class_name
                )(np.asarray(values, dtype=dtype)),
                _array_expectation,
            )

    if _available("FloatTensor"):
        _case(
            cases,
            "tensor_float32_noncontiguous_view",
            "Non-contiguous float32 tensor view with shape (3, 2)",
            lambda: sb.FloatTensor(np.arange(12, dtype=np.float32).reshape(3, 4))[
                :, ::2
            ],
            _array_expectation,
        )
        _case(
            cases,
            "tensor_float64_empty",
            "Empty float64 tensor with shape (0, 2)",
            lambda: sb.FloatTensor(np.empty((0, 2), dtype=np.float64)),
            _array_expectation,
        )

    if _available("Pcf"):
        for suffix, dtype in (("f32", np.float32), ("f64", np.float64),
                              ("i32", np.int32), ("i64", np.int64)):
            _case(
                cases,
                f"pcf_{suffix}",
                f"Standalone {suffix} piecewise constant function",
                lambda dtype=dtype: sb.Pcf(
                    np.asarray([[0, 1], [2, 3], [5, -1]], dtype=dtype)
                ),
                _array_expectation,
            )
        if _available("PcfTensor"):
            for suffix, dtype in (("f32", np.float32), ("f64", np.float64)):
                _case(
                    cases,
                    f"pcf_tensor_{suffix}",
                    f"Tensor containing two {suffix} PCFs",
                    lambda dtype=dtype: sb.PcfTensor([
                        sb.Pcf(np.asarray([[0, 1], [2, 3]], dtype=dtype)),
                        sb.Pcf(np.asarray([[0, -1], [4, 2]], dtype=dtype)),
                    ]),
                    _element_expectation,
                )
        if _available("IntPcfTensor"):
            for suffix, dtype in (("i32", np.int32), ("i64", np.int64)):
                _case(
                    cases,
                    f"pcf_tensor_{suffix}",
                    f"Tensor containing an {suffix} PCF",
                    lambda dtype=dtype: sb.IntPcfTensor([
                        sb.Pcf(np.asarray([[0, 1], [2, 3]], dtype=dtype))
                    ]),
                    _element_expectation,
                )

    if _available("PointCloudTensor"):
        for suffix, dtype in (("f32", np.float32), ("f64", np.float64)):
            factory = lambda dtype=dtype: sb.PointCloudTensor(
                np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype)
            )
            _case(
                cases,
                f"point_cloud_tensor_{suffix}",
                f"Two dense {suffix} point clouds",
                factory,
                _element_expectation,
            )

    Barcode = _barcode_class()
    BarcodeTensor = _barcode_tensor_class()
    if Barcode is not None:
        for suffix, dtype in (("f32", np.float32), ("f64", np.float64)):
            barcode_factory = lambda dtype=dtype: Barcode(
                np.asarray([[0, 1], [0.5, 2]], dtype=dtype)
            )
            _case(
                cases,
                f"barcode_{suffix}",
                f"Standalone {suffix} persistence barcode",
                barcode_factory,
                _array_expectation,
            )
            if BarcodeTensor is not None:
                _case(
                    cases,
                    f"barcode_tensor_{suffix}",
                    f"Tensor containing one {suffix} barcode",
                    lambda barcode_factory=barcode_factory, BarcodeTensor=BarcodeTensor: BarcodeTensor(
                        [barcode_factory()]
                    ),
                    _element_expectation,
                )

    matrix_types = (
        ("symmetric", "SymmetricMatrix", "SymmetricMatrixTensor",
         [[-1, 1, 2], [1, -2, 3], [2, 3, -3]]),
        ("distance", "DistanceMatrix", "DistanceMatrixTensor",
         [[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
    )
    for prefix, object_name, tensor_name, values in matrix_types:
        if not _available(object_name):
            continue
        for suffix, dtype in (("f32", np.float32), ("f64", np.float64)):
            object_factory = lambda object_name=object_name, dtype=dtype, values=values: getattr(
                sb, object_name
            )(np.asarray(values, dtype=dtype))
            _case(
                cases,
                f"{prefix}_matrix_{suffix}",
                f"Standalone 3x3 {suffix} {prefix} matrix",
                object_factory,
                _array_expectation,
            )
            if _available(tensor_name):
                _case(
                    cases,
                    f"{prefix}_matrix_tensor_{suffix}",
                    f"Tensor containing one 3x3 {suffix} {prefix} matrix",
                    lambda tensor_name=tensor_name, dtype=dtype, values=values: getattr(
                        sb, tensor_name
                    )(np.asarray([values], dtype=dtype)),
                    _element_expectation,
                )

    dtype_obj = getattr(sb, "float32", None)
    if dtype_obj is not None:
        _case(
            cases,
            "dtype_float32",
            "float32 dtype singleton metadata",
            lambda: dtype_obj,
            _singleton_expectation,
            binary=False,
        )
    return cases


def _safe_label(value):
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not label:
        raise ValueError("Version label must contain a letter or digit")
    return label


def _write_artifact(path, writer):
    with path.open("wb") as file:
        writer(file)
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="serialization golden root (default: test/golden/serialization)",
    )
    args = parser.parse_args()

    version = _distribution_version()
    output = args.output_root / _safe_label(version)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    backend = None
    try:
        sample = sb.FloatTensor(np.asarray([0], dtype=np.float32))
        backend = type(sample._data).__module__
    except Exception:
        pass

    manifest = {
        "schema_version": 1,
        "producer": {
            "package": PACKAGE_NAME,
            "version": version,
            "label": version,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "backend": backend,
            "pickle_protocol": PICKLE_PROTOCOL,
        },
        "artifacts": [],
        "unsupported": [],
    }

    for case in _cases():
        try:
            obj = case.factory()
            expected = case.expectation(obj)
        except Exception as error:
            manifest["unsupported"].append({
                "name": case.name,
                "description": case.description,
                "stage": "construct",
                "error": f"{type(error).__name__}: {error}",
            })
            continue

        formats = [("pickle", ".pkl")]
        if case.binary and hasattr(sb, "save"):
            formats.append(("binary", ".sb"))
        for serialization, suffix in formats:
            filename = case.name + suffix
            path = output / filename
            try:
                if serialization == "pickle":
                    writer = lambda file, obj=obj: pickle.dump(
                        obj, file, protocol=PICKLE_PROTOCOL
                    )
                else:
                    writer = lambda file, obj=obj: sb.save(obj, file)
                byte_count, sha256 = _write_artifact(path, writer)
            except Exception as error:
                if path.exists():
                    path.unlink()
                manifest["unsupported"].append({
                    "name": case.name,
                    "description": case.description,
                    "serialization": serialization,
                    "stage": "write",
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            manifest["artifacts"].append({
                "name": case.name,
                "description": case.description,
                "serialization": serialization,
                "file": filename,
                "bytes": byte_count,
                "sha256": sha256,
                "expected": expected,
            })

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(manifest['artifacts'])} artifacts and recorded "
        f"{len(manifest['unsupported'])} unsupported cases in {output}"
    )


if __name__ == "__main__":
    sys.exit(main())
