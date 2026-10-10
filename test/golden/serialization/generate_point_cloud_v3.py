"""Generate a pinned V3 point-cloud corpus using the installed current build."""

import hashlib
import json
import pickle
import platform
import subprocess
from pathlib import Path

import numpy as np
import stablebear as sb

from generate import _distribution_version


def main():
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    # Tests/docs may be dirty, but the producing implementation must be pinned.
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "include", "src", "stablebear"],
        cwd=root, check=True,
    )
    output = Path(__file__).resolve().parent / "0.5-pre"
    output.mkdir(exist_ok=False)
    manifest = {
        "schema_version": 1,
        "producer": {
            "package": "stablebear", "version": _distribution_version(),
            "commit": commit, "format_version": 3,
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "pickle_protocol": 4,
            "backend": type(sb.FloatTensor([0.])._data).__module__,
        },
        "artifacts": [], "unsupported": [],
    }
    for dtype in (np.float32, np.float64):
        points = sb.PointCloudTensor(np.array([[0, 1], [2, 3], [4, 5]], dtype=dtype))
        selected = points[sb.NestedTensor(sb.indices([2, 0, 2]))]
        cases = (
            ("dense", points[()], [[0, 1], [2, 3], [4, 5]], [3, 2]),
            ("selected", selected[()], [[4, 5], [0, 1], [4, 5]], [3, 2]),
            ("empty", sb.PointCloud(np.empty((0, 2), dtype=dtype)), [], [0, 2]),
        )
        for label, obj, values, shape in cases:
            expected = {
                "accepted_types": ["PointCloud"], "comparison": "array",
                "dtype": np.dtype(dtype).name, "shape": shape,
                "array": {"dtype": np.dtype(dtype).name, "shape": shape, "values": values},
            }
            name = f"point_cloud_{label}_{np.dtype(dtype).name}"
            for serialization, suffix in (("binary", "sb"), ("pickle", "pkl")):
                path = output / f"{name}.{suffix}"
                with path.open("wb") as file:
                    if serialization == "binary":
                        sb.save(obj, file)
                    else:
                        pickle.dump(obj, file, protocol=4)
                data = path.read_bytes()
                manifest["artifacts"].append({
                    "name": name, "description": f"V3 {label} standalone point cloud",
                    "serialization": serialization, "file": path.name,
                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "expected": expected,
                })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest['artifacts'])} artifacts to {output}")


if __name__ == "__main__":
    main()
