import matplotlib.pyplot as plt
import numpy as np

from stablebear import Pcf, l1_norm
from stablebear.plotting import plot as plotpcf

# Construct three PCFs of increasing complexity
f0 = Pcf(np.array([[0.0], [0.0]]))
f1 = Pcf(np.array([[0.0, 1.5], [2.0, 0.0]]))
f2 = Pcf(np.array([[0.0, 1.5, 3.0, 4.0], [2.0, 1.0, 7.0, 0.0]]))

# f0 will not show up on the plot since it's a single point
plotpcf(f0, label="f0")
plotpcf(f1, label="f1")
plotpcf(f2, label="f2")
plt.legend()

# stablebear supports l1 norm for lists of PCFs and single PCFs
print(l1_norm([f0, f1, f2]))
print(l1_norm(f1))
