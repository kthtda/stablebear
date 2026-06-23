import math
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import stablebear as sb
from stablebear import persistence as pers
from stablebear.plotting import plot as plotpcf

# The dimension of the tensor holding the point clouds
shape = (100, 20, 30)

# The minimum and maximum number of points in each point cloud
min_pcloud_points = 5
max_pcloud_points = 100

# The dimension of the points in the point clouds (in principle, the dimension could be different for each point cloud,
# but we'll keep it the same here as that's the most common case)
pcloud_dim = 4

print(f"Will compute {math.prod(shape)} barcodes...")

# Create a tensor to hold our point clouds
pclouds = sb.zeros(shape, dtype=sb.pcloud64)

# Here, we are creating random point clouds and filling in the elements of `pclouds`

# The following loop could be written as
#
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         for k in range(shape[2]):
#
# ...but wrapping a product in tqdm gives a nested loop over the same indices with progress reporting.

for i, j, k in tqdm(
    product(range(shape[0]), range(shape[1]), range(shape[2])),
    desc="Initializing data",
    total=shape[0] * shape[1] * shape[2],
):  # trange gives a progress bar
    pcloud_shape = (
        np.random.randint(min_pcloud_points, max_pcloud_points + 1),
        pcloud_dim,
    )

    # Each point cloud (i, j, k) will consist of a random number of points between `min_pcloud_points` and
    # `max_pcloud_points`, each with point dimension `pcloud_dim`
    pclouds[i, j, k] = np.random.randn(*pcloud_shape)

# compute_persistent_homology takes a tensor of point clouds and produces a tensor of barcodes (somewhat similar to
# persistence diagrams). We compute the `maxDim` first homologies, so the resulting tensor will have the same shape
# as the input tensor, with an extra dimension of size `maxDim`. To get the H_n barcode corresponding to point cloud
# (i, j, k), we would request element (i, j, k, n) from the resulting tensor.
bcs = pers.compute_persistent_homology(pclouds, maxDim=1)
print(bcs.shape)

# barcode_to_stable_rank converts all barcodes in the input to stable rank PCFs. The input and output shapes will be
# the same
sranks = pers.barcode_to_stable_rank(bcs)
print(sranks.shape)

# Plot H1 stable ranks corresponding to the point clouds in pclouds[:, 0, 0]
plotpcf(sranks[:, 0, 0, 1])
plt.show()
