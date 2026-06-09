import stablebear as sb
from stablebear.plotting import plot as plotpcf
import matplotlib.pyplot as plt
import numpy as np

Z = sb.zeros((2,), dtype=sb.pcf64)
Z[0] = sb.Pcf(np.array([[0,3], [1,2], [4,5], [6,0]]))
Z[1] = sb.Pcf(np.array([[0,2], [3,4], [4,2], [5,2], [8,3]]))

plotpcf(Z)
plt.show()
  
