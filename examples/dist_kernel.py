#!/usr/bin/env python3
"""
Created on Thu Apr  4 15:42:00 2024

@author: bwehlin
"""

import matplotlib.pyplot as plt
import numpy as np

import stablebear as sb

# import stablebear.system as system
from stablebear.plotting import plot as plotpcf

# sb.system.force_cpu(True)
sb.system.set_device_verbose(False)
sb.system.set_cuda_threshold(1)
sb.system.force_cpu(True)

f = sb.Pcf(np.array([[0, 4], [2, 3], [3, 1], [5, 0]]))
g = sb.Pcf(np.array([[0, 2], [6, 1], [7, 0]]))

print(sb.l2_kernel([f, g], verbose=False))


plotpcf(f)
plotpcf(g)
plt.show()

ps = np.linspace(1.0, 10.0)
ds = np.zeros_like(ps)
print(ds.shape)

print(sb.pdist([f, g], p=1, verbose=False))

for i, p in enumerate(ps):
    ds[i] = sb.pdist([f, g], p=p, verbose=False)[0, 1]

plt.plot(ps, ds)

# plotpcf(f)
