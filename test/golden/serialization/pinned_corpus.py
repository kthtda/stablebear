"""Write a golden corpus pinned to the current build's implementation commit.

Used by the generators that run against the current build, such as
``generate_point_cloud_v3.py``. The historical generators, ``generate.py`` and
``generate_masspcf_v1.py``, stay self-contained so they can run against older
versions.
"""

import hashlib
import json
import pickle
import platform
import subprocess
from pathlib import Path

import numpy as np
import stablebear as sb

from generate import _distribution_version


PICKLE_PROTOCOL = 4


def write_pinned_corpus(directory, cases):
    """Write each case of ``cases``, tuples of a name, a description, an
    object, and its expected value, as binary and pickle files into the new
    directory ``directory``, with a manifest naming the producing commit.

    Refuses to overwrite a corpus or to run with uncommitted implementation
    changes, so the commit in the manifest is the one that wrote the files.
    """
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    # Tests/docs may be dirty, but the producing implementation must be pinned.
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "include", "src", "stablebear"],
        cwd=root, check=True,
    )
    output = Path(__file__).resolve().parent / directory
    output.mkdir(exist_ok=False)
    manifest = {
        "schema_version": 1,
        "producer": {
            "package": "stablebear", "version": _distribution_version(),
            "commit": commit, "format_version": 3,
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "pickle_protocol": PICKLE_PROTOCOL,
            "backend": type(sb.FloatTensor([0.])._data).__module__,
        },
        "artifacts": [], "unsupported": [],
    }
    for name, description, obj, expected in cases:
        for serialization, suffix in (("binary", "sb"), ("pickle", "pkl")):
            path = output / f"{name}.{suffix}"
            with path.open("wb") as file:
                if serialization == "binary":
                    sb.save(obj, file)
                else:
                    pickle.dump(obj, file, protocol=PICKLE_PROTOCOL)
            data = path.read_bytes()
            manifest["artifacts"].append({
                "name": name, "description": description,
                "serialization": serialization, "file": path.name,
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "expected": expected,
            })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest['artifacts'])} artifacts to {output}")
