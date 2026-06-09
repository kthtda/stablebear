#!/usr/bin/env python3
"""
Created on Fri Jan 26 16:20:52 2024

@author: bwehlin
"""

import argparse
import timeit

import numpy as np
import pandas as pd

import stablebear as sb
from stablebear.random import noisy_cos

parser = argparse.ArgumentParser(prog="sbbench")

parser.add_argument("-n", "--npcfs", metavar="npcfs", type=int, help="number of PCFs")
parser.add_argument(
    "-s",
    "--seed",
    metavar="seed",
    nargs="?",
    const=0,
    default=0,
    type=int,
    help="random seed (incremented by one for each repeat)",
)
parser.add_argument(
    "-c", "--ncpus", metavar="ncpus", type=int, help="number of CPUs to use"
)
parser.add_argument(
    "-g", "--ngpus", metavar="ngpus", type=int, help="number of GPUs to use"
)
parser.add_argument(
    "-d", "--float64", action="store_true", help="use 64-bit floats instead of 32-bit"
)
parser.add_argument(
    "-o",
    "--outfile",
    metavar="outfile",
    default="times.out",
    type=str,
    help="output filename",
)
parser.add_argument(
    "-r",
    "--reps",
    metavar="reps",
    nargs="?",
    const=1,
    type=int,
    default=1,
    help="number of times to repeat the experiment",
)

args = parser.parse_args()
print(args)

sb.system.set_device_verbose(True)

if args.ncpus is not None:
    sb.system.force_cpu(True)
    sb.system.limit_cpus(args.ncpus)

if args.ngpus is not None and args.ngpus != 0:
    sb.system.force_cpu(False)
    sb.system.set_cuda_threshold(0)
    sb.system.limit_gpus(args.ngpus)

warmupPcfs = noisy_cos((10,), dtype=sb.float64 if args.float64 else sb.float32)
sb.pdist(warmupPcfs)

n_pcfs = args.npcfs
min_n_pts = 10
max_n_pts = 1000

dtype = np.float64 if args.float64 else np.float32

print(
    f"npcfs: {n_pcfs}, min_n_pts: {min_n_pts}, max_n_pts: {max_n_pts}, ncpus: {args.ncpus}, ngpus: {args.ngpus}, dtype: {dtype}, outfile: {args.outfile}"
)


def gen_pcf(n_pts):
    X = np.random.randn(2, n_pts)
    tmult = np.abs(np.random.randn(1))

    # Non-negative times in increasing order
    X[0, :] = np.abs(X[0, :])
    X[0, :] = np.sort(X[0, :])
    X[0, 0] = 0  # Start at time 0
    X[0, :] = tmult * X[0, :]

    X[1, -1] = 0  # Set last value to 0 to avoid infinite integrals

    return X


def gen_pcfs(n_pcfs, min_n_pts, max_n_pts, dtype, seed):
    np.random.seed(seed)

    n_pts = np.random.randint(min_n_pts, max_n_pts, size=(n_pcfs,))
    data = [gen_pcf(n) for n in n_pts]

    fs = sb.Array([sb.Pcf(X, dtype=dtype) for X in data])

    return fs


rows = []
for i in range(args.reps):
    seed = args.seed + i

    fs = gen_pcfs(n_pcfs, min_n_pts, max_n_pts, dtype, seed=seed)
    time = timeit.repeat(lambda: sb.pdist(fs, verbose=False), repeat=1, number=1)

    result = {
        "npcfs": n_pcfs,
        "minpts": min_n_pts,
        "maxpts": max_n_pts,
        "ncpus": args.ncpus if args.ncpus is not None else 0,
        "ngpus": args.ngpus if args.ngpus is not None else 0,
        "float64": args.float64,
        "seed": seed,
        "time": time[0],
    }

    rows.append(result)

df = pd.DataFrame(rows)
df.to_csv(args.outfile, index=False)
