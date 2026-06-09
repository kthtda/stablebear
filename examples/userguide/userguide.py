import numpy as np

import stablebear as sb

TV = np.array([[0, 4], [2, 3], [3, 1], [5, 0]])
f = sb.Pcf(TV)

print(f)

######

fMatrix = np.array(f)
print(fMatrix)

######

import matplotlib.pyplot as plt

from stablebear.plotting import plot as plotpcf

plotpcf(f)

plt.xlabel("t")
plt.ylabel("f(t)")

plt.savefig("ugfig_plot_single.png")

############# Multidimensional arrays #############

Z = sb.zeros((10, 5, 4))
print(Z.shape)

######

Z[3, :, :]  # shape = (5,4), indices [3] x [0,...,4] x [0,...,3]
Z[2:9:3, 1:, 2]  # shape = (3,4), indices [2,5,8] x [1,...,4] x [2]

######

print("Here")

from stablebear.random import noisy_cos, noisy_sin

M = 10  # Number of PCFs for each case
A = sb.zeros((2, M))

# Generate 'M' noisy sin/cos functions @ 100 resp. 15 time points each.
# Assign the sin(x) functions into the first row of 'A' and cos(x)
# into the second row.
print("1")
A[0, :] = noisy_sin((M,), n_points=100)
print("2")
A[1, :] = noisy_cos((M,), n_points=15)
print("3")

fig, ax = plt.subplots(1, 1, figsize=(6, 2))

# Plot individual noisy sin/cos functions
# stablebear can plot one-dimensional arrays (views) of PCFs in a single line
plotpcf(A[0, :], ax=ax, color="b", linewidth=0.5, alpha=0.4)
plotpcf(A[1, :], ax=ax, color="r", linewidth=0.5, alpha=0.4)

plt.show()

# Means across first axis of 'A'
Aavg = sb.mean(A, dim=1)

# Plot means
plotpcf(Aavg[0], ax=ax, color="b", linewidth=2, label="sin")
plotpcf(Aavg[1], ax=ax, color="r", linewidth=2, label="cos")

ax.set_xlabel("t [2 pi]")
ax.set_ylabel("f(t)")
ax.legend()
fig.savefig("ugfig_noisy_means.png")

############# Matrix computations #############

f1 = sb.Pcf(np.array([[0.0, 5.0], [2.0, 3.0], [5.0, 0.0]]))
f2 = sb.Pcf(np.array([[0.0, 2.0], [4.0, 7.0], [8.0, 1.0], [9.0, 0.0]]))
f3 = sb.Pcf(np.array([[0, 4], [2, 3], [3, 1], [5, 0]]))
f4 = sb.Pcf(np.array([[0, 2], [6, 1], [7, 0]]))

X = sb.Array([f1, f2, f3, f4])

print(sb.pdist(X))
print(sb.pdist(X, p=3.5))

print(sb.l2_kernel(X))
