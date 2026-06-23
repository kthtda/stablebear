# stablebear

Copyright 2024-2026 Bj&ouml;rn H. Wehlin and the stablebear contributors

## Description

**stablebear** is a Python package with its backend written in C++ and CUDA for performing highly scalable computations involving piecewise constant functions (PCFs). The primary audience is practicioners within Topological Data Analysis (TDA) wanting to do statistical analysis on invariants such as *stable rank*, *Euler characteristic curves*, *Betti curves*, and so on.

The basic objects are **numpy**-like (multidimensional) arrays of PCFs, on which we support reductions such as taking averages across a dimension, etc. For 1-D arrays, we compute Lp distance matrices and L2 kernels that can then be used as input for, e.g., clustering, SVMs, and other machine learning algorithms.

## Installing

**stablebear** can be used on all three major platforms. Pre-built wheels with CUDA support are available from PyPI:

`pip install stablebear`

For building from source (e.g. from a git checkout), a C++20 compiler is required. CUDA support is optional and auto-detected; see the [build instructions](docs/installing.rst) for details.

For any issues installing or using **stablebear**, please file an issue on this GitHub repo.

## Citing

If you use **stablebear** in your research, please use the following BibTeX citation.

```
@misc{stablebear,
      title={Massively Parallel Computation of Similarity Matrices from Piecewise Constant Invariants}, 
      author={Bj\"{o}rn H. Wehlin},
      year={2024},
      eprint={2404.07183},
      archivePrefix={arXiv},
      primaryClass={stat.CO}
}
```

## Acknowledgment

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software
Program (WASP) funded by the Knut and Alice Wallenberg Foundation.