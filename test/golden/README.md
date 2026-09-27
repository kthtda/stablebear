# Golden files

Golden data is grouped by test family so future compatibility suites can use
their own schemas and version directories.

Run every family generator with the same historical Python environment using:

```bash
python test/golden/generate.py
```

Serialization fixtures live under `serialization/<version>/`; see
[`serialization/README.md`](serialization/README.md) for generation and
validation details.
