import matplotlib.pyplot as plt
import numpy as np

import stablebear as sb
from stablebear.array import max_time
from stablebear.matrix_computations import pdist
from stablebear.plotting import plot as plotpcf
from stablebear.random import noisy_cos, noisy_sin

M = 10
A = sb.zeros((2, M))

# Generate 'M' noisy sin/cos functions @ 100 resp. 15 time points each.
# Assign the sin(x) functions into the first row of 'A' and cos(x)
# into the second row.
A[0, :] = noisy_sin((M,), n_points=100)
A[1, :] = noisy_cos((M,), n_points=15)

print("A")
print(pdist(A[1, :]))
print("B")
print(pdist(A[:, 0]))

f = sb.Pcf(np.array([[0.0, 5.0], [2.0, 3.0], [5.0, 0.0]]), dtype=sb.float32)
g = sb.Pcf(
    np.array([[0.0, 2.0], [4.0, 7.0], [8.0, 1.0], [9.0, 0.0]]), dtype=sb.float32
)
h = sb.Pcf(np.array([[0, 4], [2, 3], [3, 1], [5, 0]]), dtype=sb.float32)
k = sb.Pcf(np.array([[0, 2], [6, 1], [7, 0]]), dtype=sb.float32)

B = sb.zeros((4,))
B[0] = f
B[1] = g
B[2] = h
B[3] = k


print(pdist(B))
print(B.shape)
print(len(B.shape))

print(np.array(max_time(B, 0)))

plotpcf(B)
plt.legend()
# plotpcf(g)
plt.show()
