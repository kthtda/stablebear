import matplotlib.pyplot as plt

import stablebear as sb
from stablebear.plotting import plot as plotpcf
from stablebear.random import noisy_cos, noisy_sin

plt.style.use("science")

X = sb.zeros((10, 5, 4))
print(X[3, :, :].shape)
print(X[2:9:3, 1:, 2].shape)  # [2,5,8] x [1,...,4] x [2]

v = X._as_view()
v1 = X[3, :, :]
v2 = v1[2:3, 1:4]

print(v.data.buffer())
print(v.data.offset())
print(v.strides)

print(v1.data.buffer())
print(v1.data.offset())
print(v1.strides)

print(v2.data.buffer())
print(v2.data.offset())
print(v2.strides)

M = 10
A = sb.zeros((2, M))

# Generate 'M' noisy sin/cos functions @ 100 resp. 15 time points each.
# Assign the sin(x) functions into the first row of 'A' and cos(x)
# into the second row.
A[0, :] = noisy_sin((M,), n_points=100)
A[1, :] = noisy_cos((M,), n_points=15)

fig, ax = plt.subplots(1, 1, figsize=(6, 2))

# Plot individual noisy sin/cos functions
for j in range(A.shape[1]):
    plotpcf(A[0, j], ax=ax, color="b", linewidth=0.5, alpha=0.4)
for j in range(A.shape[1]):
    plotpcf(A[1, j], ax=ax, color="r", linewidth=0.5, alpha=0.4)

# Means across first axis of 'A'
Aavg = sb.mean(A, dim=1)

# Plot means
plotpcf(Aavg[0], ax=ax, color="b", linewidth=2, label="$\\sin(2\\pi t)$")
plotpcf(Aavg[1], ax=ax, color="r", linewidth=2, label="$\\cos(2\\pi t)$")

plt.xlabel("$t$")
plt.ylabel("$f(t)$")
plt.legend()

plt.savefig("means.png", dpi=300)

plt.show()

raise SystemExit

B = noisy_cos((10000, 500), nPoints=30)
print(B.shape)

Bbar = sb.mean(B, 1)
print(Bbar.shape)

for j in range(100):
    plotpcf(Bbar[j], color="r", linewidth=2)

plt.show()
