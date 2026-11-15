"""Generate a pinned V3 PLF corpus using the installed current build."""

import numpy as np
import stablebear as sb

from pinned_corpus import write_pinned_corpus


# Breakpoints of the PLFs in the corpus: a sloped last segment, a value that
# is not exact in binary with a flat last segment, and a single breakpoint.
SLOPED = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]
FLAT = [[0.0, 0.1], [0.25, -3.0], [1e6, 7.0], [2e6, 7.0]]
CONSTANT = [[0.0, -1.5]]


def _array(dtype, breakpoints):
    return {"dtype": dtype, "shape": [len(breakpoints), 2], "values": breakpoints}


def _cases():
    for element, tensor_dtype in ((sb.float32, sb.plf32), (sb.float64, sb.plf64)):
        name = element.name
        plf = sb.Plf(SLOPED, dtype=element)
        expected = {
            "accepted_types": ["Plf"], "comparison": "array", "dtype": name,
            "array": _array(name, SLOPED),
        }
        yield f"plf_{name}", "V3 standalone PLF", plf, expected

        grid = [[SLOPED, FLAT, CONSTANT],
                [CONSTANT, SLOPED, FLAT]]
        tensor = sb.PlfTensor([[sb.Plf(b, dtype=element) for b in row] for row in grid])
        expected = {
            "accepted_types": ["PlfTensor"], "comparison": "elements",
            "dtype": tensor_dtype.name, "shape": [2, 3],
            "elements": [
                {"index": [i, j], "array": _array(name, b)}
                for i, row in enumerate(grid) for j, b in enumerate(row)
            ],
        }
        yield f"plf_tensor_{name}", "V3 2-d tensor of PLFs", tensor, expected


if __name__ == "__main__":
    write_pinned_corpus("0.6.1-pre-plf", _cases())
