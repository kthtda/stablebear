"""Generate a pinned V3 point-cloud corpus using the installed current build."""

import numpy as np
import stablebear as sb

from pinned_corpus import write_pinned_corpus


def _cases():
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
            yield name, f"V3 {label} standalone point cloud", obj, expected


if __name__ == "__main__":
    write_pinned_corpus("0.5-pre", _cases())
