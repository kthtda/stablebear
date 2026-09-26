#!/usr/bin/env python3
"""Run every golden-test generator with one producer environment."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="root containing golden-test families (default: test/golden)",
    )
    args = parser.parse_args()

    generators = sorted(
        path for path in Path(__file__).resolve().parent.glob("*/generate.py")
        if path.is_file()
    )
    if not generators:
        raise RuntimeError("No golden-test generators found")

    for generator in generators:
        family = generator.parent.name
        command = [
            sys.executable,
            str(generator),
            "--output-root",
            str(args.golden_root / family),
        ]
        print(f"Generating {family} goldens", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
